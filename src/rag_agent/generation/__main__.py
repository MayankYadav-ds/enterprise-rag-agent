"""Generate an answer from saved RetrievedChunk evidence, without touching Qdrant."""

from __future__ import annotations

import argparse
import json
import sys
from dataclasses import asdict
from pathlib import Path

from rag_agent.generation import AnswerGenerator, GenerationConfig, GenerationError, build_provider
from rag_agent.retrieval.models import RetrievedChunk


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("question")
    parser.add_argument(
        "--evidence",
        type=Path,
        required=True,
        help="JSON list of RetrievedChunk objects (dataclasses.asdict fields)",
    )
    parser.add_argument("--json", action="store_true", help="Print structured answer JSON")
    args = parser.parse_args(argv)
    try:
        with args.evidence.open("rb") as stream:
            raw = stream.read(1048577)
        if len(raw) > 1048576:
            raise ValueError("Evidence file exceeds 1 MiB")
        records = json.loads(raw)
        if not isinstance(records, list) or len(records) > 100:
            raise ValueError("Evidence must be a list of at most 100 chunks")
        chunks = [RetrievedChunk(**item) for item in records]
        config = GenerationConfig.from_env()
        generator = AnswerGenerator(
            build_provider(config),
            max_evidence_chars=config.max_evidence_chars,
            max_chunks=config.max_chunks,
        )
        answer = generator.generate(args.question, chunks)
    except GenerationError as exc:
        print(f"Generation failed: {exc}", file=sys.stderr)
        return 2
    except (OSError, ValueError, TypeError):
        print(
            "Invalid evidence or configuration; check the input format and RAG_* settings.",
            file=sys.stderr,
        )
        return 2
    print(json.dumps(asdict(answer), ensure_ascii=False, indent=2) if args.json else answer.text)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
