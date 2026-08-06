"""
Dataset Loader for AdaptiveRAG
==============================
Loads HotpotQA (multi-hop) and PopQA (single-hop) datasets using
the HuggingFace `datasets` library for benchmarking and evaluation.
"""

from datasets import load_dataset


def load_hotpotqa(split: str = "validation") -> object:
    """Load the HotpotQA dataset (multi-hop reasoning benchmark)."""
    dataset = load_dataset("hotpot_qa", "fullwiki", split=split)
    print(f"[HotpotQA] Loaded {len(dataset)} samples from '{split}' split.")
    print(f"[HotpotQA] Sample: {dataset[0]['question']}")
    return dataset


def load_popqa(split: str = "test") -> object:
    """Load the PopQA dataset (single-hop factual benchmark)."""
    dataset = load_dataset("akariasai/PopQA", split=split)
    print(f"[PopQA]    Loaded {len(dataset)} samples from '{split}' split.")
    print(f"[PopQA]    Sample: {dataset[0]['question']}")
    return dataset


def main() -> None:
    """Entry point: load and preview both datasets."""
    print("=" * 60)
    print("AdaptiveRAG — Dataset Loader")
    print("=" * 60)
    hotpotqa = load_hotpotqa()
    print()
    popqa = load_popqa()


if __name__ == "__main__":
    main()
