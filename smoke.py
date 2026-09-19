import json
from datasets import load_dataset
from datasets.load_dataset import normalize_popqa, normalize_hotpotqa

try:
    print("Loading 2 records from PopQA (streaming)...")
    pop_ds = load_dataset("akariasai/PopQA", split="test", streaming=True)
    pop_iter = iter(pop_ds)
    p1 = normalize_popqa(next(pop_iter))
    p2 = normalize_popqa(next(pop_iter))
    print("PopQA sample 1:", json.dumps(p1)[:100])
    print("PopQA sample 2:", json.dumps(p2)[:100])

    print("Loading 2 records from HotpotQA (streaming)...")
    hot_ds = load_dataset("hotpot_qa", "distractor", split="validation", streaming=True)
    hot_iter = iter(hot_ds)
    h1 = normalize_hotpotqa(next(hot_iter))
    h2 = normalize_hotpotqa(next(hot_iter))
    print("HotpotQA sample 1:", json.dumps(h1)[:100])
    print("HotpotQA sample 2:", json.dumps(h2)[:100])
except Exception as e:
    print("Error during smoke test:", e)

print("Smoke test complete.")
