"""Toy data-builder output with a structural SFT bug PromptABI catches."""

TRAINING_ROWS = [
    {
        "id": "train-0001",
        "messages": [
            {"role": "system", "content": "Answer with short facts."},
            {"role": "user", "content": "What is PromptABI?"},
            {"role": "assistant", "content": "PromptABI verifies LLM interface contracts."},
        ],
    }
]


def build_manifest_span() -> dict[str, object]:
    return {
        "span_id": "train-0001.assistant-0",
        "target_role": "assistant",
        "rendered_region_role": "assistant",
        "start_token": 18,
        "end_token": 30,
        "region_start_token": 16,
        "region_end_token": 32,
        "loss_masked": True,
        "crosses_packing_boundary": True,
        "source_contributions": [
            {
                "source_id": "train-0001.user",
                "source_kind": "user",
                "start_token": 20,
                "end_token": 24,
                "transform": "buggy-column-shift",
            }
        ],
    }
