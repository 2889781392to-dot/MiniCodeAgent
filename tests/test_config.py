import pytest

from minicodeagent.config import LoopLimits, Settings
from minicodeagent.protocol import ToolCall


def test_environment_keeps_model_name_and_hides_key():
    settings = Settings.from_env({
        "MINICODEAGENT_API_KEY": "fake-sensitive-key",
        "MINICODEAGENT_BASE_URL": "https://gateway.example/v1/",
        "MINICODEAGENT_MODEL": "deepseek4.1f",
        "MINICODEAGENT_TIMEOUT": "25.5",
        "MINICODEAGENT_DEEPSEEK_THINKING": "disabled",
    })
    assert settings.api_key.get_secret_value() == "fake-sensitive-key"
    assert settings.model_name == "deepseek4.1f"
    assert settings.base_url == "https://gateway.example/v1/"
    assert settings.request_timeout == 25.5
    assert settings.deepseek_thinking == "disabled"
    assert "fake-sensitive-key" not in repr(settings)
    assert settings.limits == LoopLimits(max_provider_calls=12, max_tool_batches=8, max_calls_per_batch=8)


@pytest.mark.parametrize(("overrides", "message"), [
    ({"MINICODEAGENT_API_KEY": ""}, "MINICODEAGENT_API_KEY"),
    ({"MINICODEAGENT_MODEL": ""}, "MINICODEAGENT_MODEL"),
    ({"MINICODEAGENT_TIMEOUT": "nan"}, "MINICODEAGENT_TIMEOUT"),
    ({"MINICODEAGENT_TIMEOUT": "-1"}, "MINICODEAGENT_TIMEOUT"),
    ({"MINICODEAGENT_BASE_URL": "https://user:private-text@example.com"}, "MINICODEAGENT_BASE_URL"),
    ({"MINICODEAGENT_DEEPSEEK_THINKING": "enabled"}, "MINICODEAGENT_DEEPSEEK_THINKING"),
])
def test_bad_configuration_has_safe_diagnostic(overrides, message):
    with pytest.raises(ValueError, match=message) as exc:
        Settings.from_env({"MINICODEAGENT_API_KEY": "test", "MINICODEAGENT_MODEL": "model"} | overrides)
    assert "private-text" not in str(exc.value)


def test_legacy_variables_do_not_supply_required_settings_or_override_defaults():
    legacy = {
        "OPENAI_API_KEY": "legacy-key",
        "OPENAI_BASE_URL": "https://old.example",
        "OPENAI_TIMEOUT": "1",
        "MODEL_NAME": "legacy-model",
        "DEEPSEEK_THINKING": "disabled",
    }
    with pytest.raises(ValueError, match=r"^Set MINICODEAGENT_API_KEY before running MiniCodeAgent\.$"):
        Settings.from_env(legacy)
    settings = Settings.from_env(legacy | {
        "MINICODEAGENT_API_KEY": "project-key",
        "MINICODEAGENT_MODEL": "project-model",
    })
    assert settings.api_key.get_secret_value() == "project-key"
    assert settings.model_name == "project-model"
    assert settings.base_url is None
    assert settings.request_timeout == 60.0
    assert settings.deepseek_thinking is None


def test_call_preserves_provider_identity_and_raw_arguments():
    call = ToolCall(provider_call_id="provider-id", name="read_file", arguments_raw="not json")
    assert call.provider_call_id == "provider-id"
    assert call.arguments_raw == "not json"
    assert call.wire_id == f"call_{call.call_id.hex}"
