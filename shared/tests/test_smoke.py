def test_shared_package_imports():
    import shared  # noqa: F401
    from shared import config, db  # noqa: F401
