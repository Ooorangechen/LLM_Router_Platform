"""P1 task 3.5: _load_config error handling."""

import pytest

import main


@pytest.mark.parametrize("text", ["- not-a-mapping\n", "api: [unclosed\n"])
def test_custom_config_rejects_invalid_yaml_documents(tmp_path, text):
    path = tmp_path / "invalid.yaml"
    path.write_text(text, encoding="utf-8")
    with pytest.raises(SystemExit):
        main.LLMRouterPlatform(path)
