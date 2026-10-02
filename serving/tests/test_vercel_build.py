"""`deploy/vercel/build.py`: copia el código del monorepo y solo usa datos con el
sha256 esperado (un paquete alterado no se despliega)."""
import hashlib
import importlib.util
import tarfile
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[2]


def _load_build():
    spec = importlib.util.spec_from_file_location("vercel_build", ROOT / "deploy/vercel/build.py")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def _bundle(tmp_path: Path) -> Path:
    (tmp_path / "src" / "processed" / "dem").mkdir(parents=True)
    (tmp_path / "src" / "processed" / "dem" / "dem.tif").write_bytes(b"fake")
    archive = tmp_path / "bundle.tar.gz"
    with tarfile.open(archive, "w:gz") as tar:
        tar.add(tmp_path / "src" / "processed", arcname="processed")
    return archive


def test_fetch_data_extracts_a_bundle_with_the_right_sha256(tmp_path):
    build = _load_build()
    archive = _bundle(tmp_path)
    digest = hashlib.sha256(archive.read_bytes()).hexdigest()
    processed = build.fetch_data(str(archive), digest, tmp_path / "data")
    assert (processed / "dem" / "dem.tif").read_bytes() == b"fake"


def test_fetch_data_refuses_a_bundle_with_another_sha256(tmp_path):
    build = _load_build()
    archive = _bundle(tmp_path)
    with pytest.raises(SystemExit, match="sha256"):
        build.fetch_data(str(archive), "0" * 64, tmp_path / "data")
    assert not (tmp_path / "data").exists()


def test_vendor_sources_copies_every_package_and_the_web_without_tests(tmp_path):
    build = _load_build()
    build.vendor_sources(ROOT, tmp_path / "_vendor")
    for package in build.PACKAGES:
        assert (tmp_path / "_vendor" / package / "src" / package / "__init__.py").exists()
    assert (tmp_path / "_vendor" / "serving" / "web" / "static" / "app.js").exists()
    assert not list((tmp_path / "_vendor").rglob("tests"))


def test_data_bundle_manifest_points_to_a_pinned_release_asset():
    import json

    manifest = json.loads((ROOT / "deploy/vercel/data-bundle.json").read_text())
    assert manifest["url"].startswith("https://github.com/")
    assert "/releases/download/" in manifest["url"]
    assert len(manifest["sha256"]) == 64
