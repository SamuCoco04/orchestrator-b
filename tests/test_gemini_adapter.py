import importlib
import sys
import types


class _FakeModels:
    def __init__(self) -> None:
        self.generate_calls = []

    def list(self):
        return [types.SimpleNamespace(name="models/gemini-2.5-flash")]

    def generate_content(self, **kwargs):
        self.generate_calls.append(kwargs.get("model"))
        return types.SimpleNamespace(text='{"ok": true}')


class _FakeClient:
    def __init__(self, api_key: str) -> None:
        self.models = _FakeModels()


def test_selects_flash_when_pro_missing(monkeypatch):
    fake_google = types.ModuleType("google")
    fake_google.genai = types.SimpleNamespace(Client=_FakeClient)
    monkeypatch.setitem(sys.modules, "google", fake_google)

    monkeypatch.setenv("GEMINI_API_KEY", "test-key")
    monkeypatch.delenv("ORCH_GEMINI_MODELS", raising=False)

    module = importlib.import_module("src.adapters.gemini_adapter")
    GeminiAdapter = module.GeminiAdapter

    adapter = GeminiAdapter()
    text = adapter.generate("hello")

    assert text == '{"ok": true}'
    assert adapter.client.models.generate_calls == ["gemini-2.5-flash"]
    diagnostics = adapter.get_diagnostics()
    assert diagnostics["selected_model"] == "gemini-2.5-flash"
