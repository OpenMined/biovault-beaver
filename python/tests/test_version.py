import beaver


def test_version():
    assert beaver.__version__ is not None
    assert isinstance(beaver.__version__, str)
