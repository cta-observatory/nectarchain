def test_version():
    from nectarchain import __version__

    assert __version__ != '0.0.0'