from importlib.metadata import version

import minicodeagent


def test_package_is_installed():
    assert minicodeagent.__name__ == "minicodeagent"
    assert version("minicodeagent") == "0.1.0"
