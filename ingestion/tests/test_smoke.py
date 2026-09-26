def test_ingestion_subpackages_import():
    from ingestion import dem, era5, firms, sentinel2, worldcover  # noqa: F401
