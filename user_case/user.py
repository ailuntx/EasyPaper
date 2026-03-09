"""
EasyPaper SDK Usage Demo

Demonstrates both one-shot and streaming paper generation.
Before running, create a config.yaml with your API keys (see config.example.yaml).
"""
import asyncio
from pathlib import Path

from easypaper import EasyPaper, PaperMetaData, EventType


async def demo_stream():
    """Streaming mode: observe every generation phase in real time."""

    config_path = Path(__file__).parent / "config.yaml"
    ep = EasyPaper(config_path=str(config_path))

    metadata = PaperMetaData(
        title="Attention Is All You Need: A Revisit",
        idea_hypothesis=(
            "Transformer architectures with pure self-attention mechanisms "
            "can replace recurrent and convolutional layers entirely, achieving "
            "superior performance on sequence-to-sequence tasks while being "
            "more parallelizable and requiring significantly less training time."
        ),
        method=(
            "We propose the Transformer, a model architecture eschewing recurrence "
            "and instead relying entirely on an attention mechanism to draw global "
            "dependencies between input and output. The Transformer allows for "
            "significantly more parallelization and can reach a new state of the "
            "art in translation quality."
        ),
        data=(
            "We train on the standard WMT 2014 English-German dataset consisting "
            "of about 4.5 million sentence pairs. We also evaluate on the WMT 2014 "
            "English-French translation task."
        ),
        experiments=(
            "On the WMT 2014 English-to-German translation task, the big transformer "
            "model outperforms the best previously reported models including ensembles "
            "by more than 2.0 BLEU, establishing a new state-of-the-art BLEU score "
            "of 28.4. On the WMT 2014 English-to-French translation task, our model "
            "achieves a new single-model state-of-the-art BLEU score of 41.0."
        ),
    )

    print("=" * 60)
    print("  EasyPaper — Streaming Generation Demo")
    print("=" * 60)
    print()

    async for event in ep.generate_stream(metadata):
        if event.event_type == EventType.PHASE_START:
            print(f"▶ [{event.phase}] {event.message}")

        elif event.event_type == EventType.PHASE_COMPLETE:
            print(f"✓ [{event.phase}] {event.message}")
            print()

        elif event.event_type == EventType.SECTION_COMPLETE:
            word_count = event.data.get("word_count", "?") if event.data else "?"
            print(f"  ✎ Section done: {event.phase}  ({word_count} words)")

        elif event.event_type == EventType.PROGRESS:
            print(f"  … {event.message}")

        elif event.event_type == EventType.WARNING:
            print(f"  ⚠ {event.message}")

        elif event.event_type == EventType.ERROR:
            print(f"  ✗ ERROR: {event.message}")

        elif event.event_type == EventType.COMPLETE:
            print("-" * 60)
            result = event.data.get("result") if event.data else None
            if result:
                print(f"Paper: {result.get('paper_title', 'N/A')}")
                print(f"Status: {result.get('status', 'N/A')}")
                print(f"Total words: {result.get('total_word_count', 'N/A')}")
                print(f"Sections: {len(result.get('sections', []))}")
                if result.get("output_path"):
                    print(f"Output: {result['output_path']}")
            print("Done!")


async def demo_oneshot():
    """One-shot mode: generate and get the final result directly."""

    config_path = Path(__file__).parent / "config.yaml"
    ep = EasyPaper(config_path=str(config_path))

    metadata = PaperMetaData(
        title="Attention Is All You Need: A Revisit",
        idea_hypothesis="Transformer with pure self-attention can replace RNNs.",
        method="Multi-head attention with positional encoding.",
        data="WMT 2014 En-De and En-Fr translation benchmarks.",
        experiments="BLEU 28.4 on En-De, 41.0 on En-Fr.",
    )

    print("Generating paper (one-shot)…")
    result = await ep.generate(metadata)
    print(f"Status: {result.status}")
    print(f"Sections: {len(result.sections)}")
    for sec in result.sections:
        print(f"  - {sec.section_type}: {sec.word_count} words [{sec.status}]")


if __name__ == "__main__":
    asyncio.run(demo_stream())
