"""Report transport and prompt assembly without network calls."""
import importlib.util
import json
import sys
from pathlib import Path
from unittest.mock import MagicMock

import pytest


@pytest.fixture
def reports(monkeypatch):
    monkeypatch.setenv("OPENAI_BASE_URL", "https://provider.example/v1/")
    monkeypatch.setenv("OPENAI_API_KEY", "test-key")
    monkeypatch.setenv("EE_REPORT_MODEL", "remote-model")
    path = Path(__file__).parents[1] / "scripts" / "llm_reports.py"
    spec = importlib.util.spec_from_file_location("report_script", path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def test_remote_chat_url_model_and_auth(reports, monkeypatch):
    response = MagicMock()
    response.__enter__.return_value.read.return_value = json.dumps(
        {"choices": [{"message": {"content": "report"}}]}
    ).encode()
    transport = MagicMock(return_value=response)
    monkeypatch.setattr(reports.urllib.request, "urlopen", transport)
    assert reports.chat([{"role": "user", "content": "summary"}]) == "report"
    request = transport.call_args.args[0]
    assert request.full_url == "https://provider.example/v1/chat/completions"
    assert request.get_header("Authorization") == "Bearer test-key"
    assert json.loads(request.data)["model"] == "remote-model"


@pytest.mark.parametrize("discover_model", [False, True])
def test_experience_model_configuration(reports, monkeypatch, tmp_path, discover_model):
    log = tmp_path / "experiment_log.json"
    log.write_text("{}")
    output = tmp_path / "nested" / "training.md"
    monkeypatch.setattr(sys, "argv", ["llm_reports", "--mode", "experience",
                                       "--log", str(log), "--output", str(output)])
    monkeypatch.setattr(reports, "build_payload", lambda log, mode: {})
    monkeypatch.setattr(reports, "chat", lambda messages: "training report")
    response = MagicMock()
    response.__enter__.return_value.read.return_value = b'{"data":[{"id":"discovered"}]}'
    transport = MagicMock(return_value=response)
    monkeypatch.setattr(reports.urllib.request, "urlopen", transport)
    if discover_model:
        reports.MODEL_ID = ""
    reports.main()
    assert output.read_text() == "training report"
    if discover_model:
        assert reports.MODEL_ID == "discovered"
        assert transport.call_args.args[0].get_header("Authorization") == "Bearer test-key"
    else:
        transport.assert_not_called()


def test_final_prompt_includes_generated_plot_paths(reports, monkeypatch, tmp_path):
    log = tmp_path / "experiment_log.json"
    log.write_text("{}")
    monkeypatch.setattr(sys, "argv", ["llm_reports", "--mode", "final", "--log", str(log)])
    monkeypatch.setattr(reports, "generate_window_plots",
                        lambda log, path: [{"file": "plots/window_1.png"}])
    monkeypatch.setattr(reports, "build_payload", lambda log, mode: log)
    messages = []
    monkeypatch.setattr(reports, "chat", lambda value: messages.extend(value) or "final report")
    reports.main()
    assert '"file": "plots/window_1.png"' in messages[1]["content"]
    assert (tmp_path / "final_report.md").read_text() == "final report"
