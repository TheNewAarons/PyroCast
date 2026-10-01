"""Robustez de la ingesta (revisión de endurecimiento): timeouts,
reintentos con backoff, agotamiento de cuota y mensajes claros. Cero red
real (`responses` + fakes)."""
import pytest
import requests
import responses
from ingestion.dem.client import DemDownloadError, TileNotFoundError, download_tile
from ingestion.dem.client import download_tile as dem_download
from ingestion.dem.pipeline import build_dem
from ingestion.dem.tiles import tile_url as dem_tile_url
from ingestion.era5.client import (
    Era5Client,
    Era5QuotaExceededError,
    Era5RequestFailedError,
)
from ingestion.firms.client import FirmsApiError, FirmsClient, FirmsQuotaExceededError
from ingestion.resilience import (
    IngestionError,
    QuotaExceededError,
    SourceUnavailableError,
    get_with_retry,
    handle_errors,
    redact,
)
from ingestion.sentinel2.client import (
    Sentinel2AuthError,
    Sentinel2Client,
    Sentinel2QuotaExceededError,
)
from ingestion.worldcover.client import WorldCoverDownloadError
from ingestion.worldcover.client import download_tile as wc_download
from ingestion.worldcover.tiles import tile_url as wc_tile_url

URL = "https://example.test/file.tif"
KEY = "Copernicus_DSM_COG_10_S37_00_W072_00_DEM"


def _sleeps():
    calls: list[float] = []
    return calls, calls.append


# ---------------------------------------------------------------- get_with_retry
@responses.activate
def test_get_with_retry_retries_5xx_with_exponential_backoff_then_succeeds():
    responses.add(responses.GET, URL, status=503)
    responses.add(responses.GET, URL, status=502)
    responses.add(responses.GET, URL, body=b"ok", status=200)
    sleeps, sleep_fn = _sleeps()
    response = get_with_retry(requests.Session(), URL, source="Fuente X", sleep_fn=sleep_fn,
                              backoff_base=1.0)
    assert response.content == b"ok"
    assert sleeps == [1.0, 2.0]


@responses.activate
def test_get_with_retry_exhausted_429_raises_quota_error_naming_the_source():
    for _ in range(3):
        responses.add(responses.GET, URL, status=429, headers={"Retry-After": "7"})
    sleeps, sleep_fn = _sleeps()
    with pytest.raises(QuotaExceededError) as exc:
        get_with_retry(requests.Session(), URL, source="Fuente X", max_retries=2,
                       sleep_fn=sleep_fn)
    assert "Fuente X" in str(exc.value) and "cuota" in str(exc.value).lower()
    assert exc.value.source == "Fuente X" and exc.value.hint
    assert sleeps == [7.0, 7.0]  # respeta Retry-After


@responses.activate
def test_get_with_retry_caps_retry_after():
    responses.add(responses.GET, URL, status=429, headers={"Retry-After": "86400"})
    responses.add(responses.GET, URL, body=b"ok", status=200)
    sleeps, sleep_fn = _sleeps()
    get_with_retry(requests.Session(), URL, source="X", sleep_fn=sleep_fn, retry_after_cap=30.0)
    assert sleeps == [30.0]


@responses.activate
def test_get_with_retry_wraps_transport_errors_never_leaks_requests_exceptions():
    for _ in range(3):
        responses.add(responses.GET, URL, body=requests.ConnectionError("boom"))
    with pytest.raises(SourceUnavailableError) as exc:
        get_with_retry(requests.Session(), URL, source="Fuente X", max_retries=2,
                       sleep_fn=lambda _s: None)
    assert "Fuente X" in str(exc.value) and "boom" in str(exc.value)


@responses.activate
def test_get_with_retry_passes_a_connect_and_read_timeout():
    responses.add(responses.GET, URL, body=b"ok")
    get_with_retry(requests.Session(), URL, source="X", timeout=(3.0, 9.0))
    assert responses.calls[0].request.req_kwargs["timeout"] == (3.0, 9.0)


@responses.activate
def test_get_with_retry_returns_non_retryable_status_for_the_caller_to_judge():
    responses.add(responses.GET, URL, status=404)
    assert get_with_retry(requests.Session(), URL, source="X").status_code == 404
    assert len(responses.calls) == 1  # un 404 no se reintenta


def test_redact_removes_every_secret_and_ignores_empty_ones():
    text = redact("key=abc123 and abc123 again", ["abc123", ""])
    assert text == "key=<REDACTED> and <REDACTED> again"


# ------------------------------------------------------------------- DEM / WorldCover
@responses.activate
def test_dem_retries_a_transient_503_and_then_downloads(tmp_path, monkeypatch):
    monkeypatch.setattr("ingestion.resilience.time.sleep", lambda _s: None)
    responses.add(responses.GET, dem_tile_url(KEY), status=503)
    responses.add(responses.GET, dem_tile_url(KEY), body=b"tif", status=200)
    assert dem_download(KEY, tmp_path / "t.tif").read_bytes() == b"tif"


@responses.activate
def test_dem_404_is_a_tile_not_found_error_and_a_dem_download_error(tmp_path):
    responses.add(responses.GET, dem_tile_url(KEY), status=404)
    with pytest.raises(TileNotFoundError):
        download_tile(KEY, tmp_path / "t.tif")
    assert issubclass(TileNotFoundError, DemDownloadError)
    assert issubclass(DemDownloadError, IngestionError)


@responses.activate
def test_dem_network_failure_is_source_unavailable_with_a_hint(tmp_path, monkeypatch):
    monkeypatch.setattr("ingestion.resilience.time.sleep", lambda _s: None)
    for _ in range(6):
        responses.add(responses.GET, dem_tile_url(KEY), body=requests.Timeout("lento"))
    with pytest.raises(SourceUnavailableError) as exc:
        dem_download(KEY, tmp_path / "t.tif")
    assert "Copernicus DEM" in str(exc.value)
    assert not (tmp_path / "t.tif").exists()


@responses.activate
def test_dem_connection_cut_mid_body_is_retried_then_reported_not_cached(tmp_path, monkeypatch):
    monkeypatch.setattr("ingestion.resilience.time.sleep", lambda _s: None)
    for _ in range(6):
        responses.add(responses.GET, dem_tile_url(KEY), body=b"abc", status=200,
                      headers={"Content-Length": "100"}, auto_calculate_content_length=False)
    with pytest.raises(SourceUnavailableError, match="sin respuesta"):
        dem_download(KEY, tmp_path / "t.tif")
    assert not (tmp_path / "t.tif").exists()


def test_dem_content_length_mismatch_is_rejected_even_if_the_http_layer_missed_it(tmp_path):
    class _Resp:
        status_code = 200
        headers = {"Content-Length": "100"}
        content = b"abc"

    class _Session:
        def get(self, url, timeout):
            return _Resp()

    with pytest.raises(DemDownloadError, match="truncad"):
        dem_download(KEY, tmp_path / "t.tif", session=_Session())
    assert not (tmp_path / "t.tif").exists()


def test_build_dem_tolerates_only_missing_tiles_not_network_failures(tmp_path):
    bbox = (-72.5, -37.5, -71.5, -36.5)  # varios tiles de 1°

    def only_404s(key, dest):
        raise TileNotFoundError(f"404 simulado para {key}")

    with pytest.raises(DemDownloadError, match="Ninguno"):
        build_dem(bbox, 250, "EPSG:32719", tmp_path / "raw", tmp_path / "cache", only_404s)

    calls = []

    def network_down_on_second(key, dest):
        calls.append(key)
        if len(calls) == 2:
            raise SourceUnavailableError("Copernicus DEM", "sin red", hint="reintentar")
        raise TileNotFoundError("404")

    with pytest.raises(SourceUnavailableError):
        build_dem(bbox, 250, "EPSG:32719", tmp_path / "raw", tmp_path / "cache",
                  network_down_on_second)


@responses.activate
def test_worldcover_has_same_retry_and_error_taxonomy(tmp_path, monkeypatch):
    monkeypatch.setattr("ingestion.resilience.time.sleep", lambda _s: None)
    url = wc_tile_url("N36W075")
    responses.add(responses.GET, url, status=500)
    responses.add(responses.GET, url, body=b"wc", status=200)
    assert wc_download("N36W075", tmp_path / "w.tif").read_bytes() == b"wc"
    assert issubclass(WorldCoverDownloadError, IngestionError)


# --------------------------------------------------------------------------- FIRMS
@responses.activate
def test_firms_quota_exhaustion_is_explicit():
    base = "https://firms.modaps.eosdis.nasa.gov/api/area/csv/SECRETKEY/VIIRS_SNPP_NRT"
    responses.add(responses.GET, f"{base}/-73,-39,-71,-36/1", status=429)
    client = FirmsClient("SECRETKEY", max_retries=1, sleep_fn=lambda _s: None,
                         min_request_interval_seconds=0)
    with pytest.raises(FirmsQuotaExceededError) as exc:
        client.fetch_area_csv((-73, -39, -71, -36), "VIIRS_SNPP_NRT", 1)
    text = str(exc.value)
    assert "5000" in text and "10 min" in text and "SECRETKEY" not in text
    assert isinstance(exc.value, FirmsApiError) and isinstance(exc.value, QuotaExceededError)


@responses.activate
def test_firms_requests_use_connect_and_read_timeouts():
    base = "https://firms.modaps.eosdis.nasa.gov/api/area/csv/K/VIIRS_SNPP_NRT"
    responses.add(responses.GET, f"{base}/-73,-39,-71,-36/1", body="")
    FirmsClient("K", min_request_interval_seconds=0).fetch_area_csv(
        (-73, -39, -71, -36), "VIIRS_SNPP_NRT", 1
    )
    timeout = responses.calls[0].request.req_kwargs["timeout"]
    assert isinstance(timeout, tuple) and len(timeout) == 2


# ---------------------------------------------------------------------------- ERA5
class _Remote:
    def __init__(self, fail_download_times=0, error=None):
        self.status = "successful"
        self.fail = fail_download_times
        self.error = error
        self.downloads = 0

    def update(self):
        pass

    def download(self, target):
        self.downloads += 1
        from pathlib import Path

        if self.fail > 0:
            self.fail -= 1
            Path(target).write_bytes(b"parcial")  # un archivo a medias
            raise (self.error or requests.ConnectionError("se cortó"))
        Path(target).write_bytes(b"netcdf")


class _CdsClient:
    def __init__(self, retrieve_errors=(), remote=None):
        self.retrieve_errors = list(retrieve_errors)
        self.remote = remote or _Remote()
        self.retrieve_calls = 0

    def __call__(self, url, key, wait_until_complete):
        return self

    def retrieve(self, dataset, request):
        self.retrieve_calls += 1
        if self.retrieve_errors:
            raise self.retrieve_errors.pop(0)
        return self.remote


def _era5(client, **kw):
    return Era5Client("https://cds.test/api", "CDSSECRET", client_factory=client,
                      sleep_fn=lambda _s: None, **kw)


def test_era5_retries_transient_errors_on_retrieve(tmp_path):
    cds = _CdsClient(retrieve_errors=[requests.ConnectionError("x"), requests.Timeout("y")])
    out = _era5(cds).download_hourly("d", {}, tmp_path / "o.nc", poll_interval_seconds=0,
                                     sleep_fn=lambda _s: None)
    assert out.read_bytes() == b"netcdf" and cds.retrieve_calls == 3


def test_era5_quota_error_is_explicit_and_not_retried(tmp_path):
    cds = _CdsClient(retrieve_errors=[Exception("Too many queued requests: limit reached")])
    with pytest.raises(Era5QuotaExceededError) as exc:
        _era5(cds).download_hourly("d", {}, tmp_path / "o.nc", sleep_fn=lambda _s: None)
    assert "cuota" in str(exc.value).lower() and cds.retrieve_calls == 1
    assert isinstance(exc.value, QuotaExceededError)


def test_era5_fatal_error_is_wrapped_and_never_contains_the_key(tmp_path):
    cds = _CdsClient(retrieve_errors=[Exception("401 Unauthorized for key CDSSECRET")])
    with pytest.raises(Era5RequestFailedError) as exc:
        _era5(cds).download_hourly("d", {}, tmp_path / "o.nc", sleep_fn=lambda _s: None)
    assert "CDSSECRET" not in str(exc.value) and "401" in str(exc.value)


def test_era5_interrupted_download_leaves_no_partial_target_and_is_retried(tmp_path):
    cds = _CdsClient(remote=_Remote(fail_download_times=1))
    target = tmp_path / "o.nc"
    _era5(cds).download_hourly("d", {}, target, poll_interval_seconds=0,
                               sleep_fn=lambda _s: None)
    assert target.read_bytes() == b"netcdf" and not (tmp_path / "o.nc.part").exists()
    cds2 = _CdsClient(remote=_Remote(fail_download_times=99))
    target2 = tmp_path / "p.nc"
    with pytest.raises(IngestionError):
        _era5(cds2, max_retries=1).download_hourly("d", {}, target2, sleep_fn=lambda _s: None)
    assert not target2.exists()


def test_era5_pipeline_resumes_by_skipping_already_downloaded_chunks(tmp_path):
    import datetime as dt

    from ingestion.era5.pipeline import fetch_daily_era5

    class Spy:
        def __init__(self):
            self.calls = []

        def download_hourly(self, dataset, request, path, timeout_seconds):
            self.calls.append(path.name)
            raise SourceUnavailableError("ERA5-Land", "sin red", hint="reintentar")

    raw = tmp_path / "raw"
    raw.mkdir()
    spy = Spy()
    start, end = dt.date(2026, 1, 20), dt.date(2026, 2, 10)  # dos meses
    # simula que el tramo de enero ya se bajó en una corrida anterior
    from ingestion.era5.cache import cache_key_for
    from ingestion.era5.client import ERA5_VARIABLES

    key = cache_key_for(start, end, ERA5_VARIABLES)
    (raw / f"era5_hourly_{key}_2026-01-20_2026-01-31.nc").write_bytes(b"x")
    with pytest.raises(SourceUnavailableError):
        fetch_daily_era5((-73, -39, -71, -36), start, end, spy, raw, tmp_path / "cache")
    assert len(spy.calls) == 1 and "2026-02" in spy.calls[0]  # enero se saltó


# ----------------------------------------------------------------------- Sentinel-2
class _OpenEoError(Exception):
    def __init__(self, message, http_status_code=None, code=None):
        super().__init__(message)
        self.http_status_code = http_status_code
        self.code = code


class _Conn:
    def __init__(self, auth_error=None, download_errors=()):
        self.auth_error = auth_error
        self.download_errors = list(download_errors)
        self.downloads = 0

    def authenticate_oidc_client_credentials(self, client_id, client_secret):
        if self.auth_error:
            raise self.auth_error

    def load_collection(self, *a, **k):
        return self

    def band(self, name):
        return self

    def __eq__(self, other):
        return self

    def __or__(self, other):
        return self

    def resample_cube_spatial(self, cube):
        return self

    def mask(self, m):
        return self

    def filter_bands(self, b):
        return self

    def reduce_dimension(self, **k):
        return self

    def download(self, path, format):
        from pathlib import Path

        self.downloads += 1
        if self.download_errors:
            raise self.download_errors.pop(0)
        Path(path).write_bytes(b"tif")


def _s2(conn):
    return Sentinel2Client("my-id", "SUPERSECRET", connect_fn=lambda url: conn,
                           sleep_fn=lambda _s: None)


def test_sentinel2_auth_failure_is_explicit_and_hides_the_secret(tmp_path):
    conn = _Conn(auth_error=_OpenEoError("invalid client SUPERSECRET", http_status_code=401))
    with pytest.raises(Sentinel2AuthError) as exc:
        _s2(conn)
    assert "SUPERSECRET" not in str(exc.value)
    assert "COPERNICUS_DATASPACE_CLIENT_ID" in exc.value.hint


def test_sentinel2_credit_exhaustion_is_a_quota_error(tmp_path):
    conn = _Conn(download_errors=[_OpenEoError("Not enough credits", http_status_code=402)])
    with pytest.raises(Sentinel2QuotaExceededError):
        _s2(conn).fetch_monthly_composite((-73, -39, -71, -36), 2026, 1, tmp_path / "o.tif")


def test_sentinel2_retries_transient_download_errors(tmp_path):
    conn = _Conn(download_errors=[requests.ConnectionError("x"),
                                  _OpenEoError("bad gateway", http_status_code=502)])
    out = _s2(conn).fetch_monthly_composite((-73, -39, -71, -36), 2026, 1, tmp_path / "o.tif")
    assert out.read_bytes() == b"tif" and conn.downloads == 3
    assert not (tmp_path / "o.tif.part").exists()


# ------------------------------------------------------------------------------ CLI
def test_cli_prints_a_clear_message_and_exit_code_instead_of_a_traceback():
    import typer
    from typer.testing import CliRunner

    app = typer.Typer()

    @app.command()
    @handle_errors
    def boom() -> None:
        raise SourceUnavailableError("Copernicus DEM", "timeout tras 4 reintentos",
                                     hint="Reintenta; los tiles ya bajados quedan en caché.")

    @app.command()
    @handle_errors
    def quota() -> None:
        raise QuotaExceededError("NASA FIRMS", "cuota agotada", hint="Espera 10 minutos.")

    runner = CliRunner()
    result = runner.invoke(app, ["boom"])
    assert result.exit_code == 1
    assert "[Copernicus DEM]" in result.output and "timeout tras 4 reintentos" in result.output
    assert "Reintenta" in result.output and "Traceback" not in result.output
    assert runner.invoke(app, ["quota"]).exit_code == 3


def test_ingestion_cli_wraps_every_source_command():
    from ingestion.cli import app

    for command in app.registered_commands:
        assert getattr(command.callback, "__wrapped__", None) is not None, command.name
    assert app.pretty_exceptions_show_locals is False  # los locals pueden contener credenciales


def test_dem_cli_failure_exits_cleanly(monkeypatch, tmp_path):
    from ingestion.cli import app
    from typer.testing import CliRunner

    for k, v in {"FIRMS_MAP_KEY": "x", "CDS_API_URL": "x", "CDS_API_KEY": "x",
                 "COPERNICUS_DATASPACE_CLIENT_ID": "x", "COPERNICUS_DATASPACE_CLIENT_SECRET": "x",
                 "POSTGRES_HOST": "x", "POSTGRES_PORT": "1", "POSTGRES_DB": "x",
                 "POSTGRES_USER": "x", "POSTGRES_PASSWORD": "x"}.items():
        monkeypatch.setenv(k, v)
    from shared.config import get_settings

    get_settings.cache_clear()

    def failing(**kwargs):
        raise SourceUnavailableError("Copernicus DEM", "sin red", hint="reintentar")

    monkeypatch.setattr("ingestion.dem.cli.build_dem", failing)
    result = CliRunner().invoke(app, ["dem"])
    get_settings.cache_clear()
    assert result.exit_code == 1 and "[Copernicus DEM]" in result.output
