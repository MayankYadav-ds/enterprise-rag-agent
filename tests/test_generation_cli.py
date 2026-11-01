"""Generation CLI tests with injected providers and temporary evidence files."""

import json

import pytest

from rag_agent.generation import ProviderError
from rag_agent.generation import __main__ as cli


def evidence_file(tmp_path):
    path = tmp_path / "evidence.json"
    path.write_text(
        json.dumps(
            [
                dict(
                    chunk_id="c",
                    doc_id="d",
                    source_file="report.pdf",
                    page_start=2,
                    page_end=2,
                    section_heading=None,
                    subheading=None,
                    chunk_type="text",
                    text="Revenue was ten.",
                    score=1.0,
                    rank=1,
                    retriever="bm25",
                )
            ]
        ),
        encoding="utf-8",
    )
    return path


@pytest.mark.parametrize("structured", [True, False])
def test_cli_with_mocked_llm(tmp_path, monkeypatch, capsys, structured):
    class MockProvider:
        def complete(self, messages, *, schema):
            return json.dumps(
                {
                    "status": "answered",
                    "selections": [{"evidence_id": "E1", "quote": "Revenue was ten."}],
                }
            )

    monkeypatch.setenv("RAG_LLM_MODEL", "mock")
    monkeypatch.setattr(cli, "build_provider", lambda config: MockProvider())
    args = ["What revenue?", "--evidence", str(evidence_file(tmp_path))]
    if structured:
        args.append("--json")
    assert cli.main(args) == 0
    output = capsys.readouterr().out
    if structured:
        result = json.loads(output)
        assert result["status"] == "answered"
        assert result["excerpts"][0]["citation"] == "[report.pdf, p. 2]"
    else:
        assert '"Revenue was ten." [report.pdf, p. 2]' in output


@pytest.mark.parametrize(
    "contents",
    ["{}", "not json", "[{}]", "x" * 1048577],
    ids=["object", "invalid_json", "missing_fields", "oversized"],
)
def test_invalid_evidence_exits_cleanly(tmp_path, capsys, contents):
    path = tmp_path / "bad.json"
    path.write_text(contents, encoding="utf-8")
    assert cli.main(["question", "--evidence", str(path)]) == 2
    assert "Invalid evidence or configuration" in capsys.readouterr().err


def test_missing_model_configuration(tmp_path, monkeypatch, capsys):
    monkeypatch.delenv("RAG_LLM_MODEL", raising=False)
    assert cli.main(["question", "--evidence", str(evidence_file(tmp_path))]) == 2
    assert "configuration" in capsys.readouterr().err


def test_provider_failure_is_nonzero(tmp_path, monkeypatch, capsys):
    class Unavailable:
        def complete(self, messages, *, schema):
            raise ProviderError("service unavailable")

    monkeypatch.setenv("RAG_LLM_MODEL", "mock")
    monkeypatch.setattr(cli, "build_provider", lambda config: Unavailable())
    assert cli.main(["question", "--evidence", str(evidence_file(tmp_path))]) == 2
    assert "Generation failed" in capsys.readouterr().err


def test_help_needs_no_model(capsys):
    with pytest.raises(SystemExit) as caught:
        cli.main(["--help"])
    assert caught.value.code == 0
    assert "--evidence" in capsys.readouterr().out
