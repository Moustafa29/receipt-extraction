"""Does valid_line cover the whole page, or only the labelled fields?"""
import json
from datasets import load_dataset


def main() -> None:
    ds = load_dataset("naver-clova-ix/cord-v2")
    sample = ds["train"][0]
    gt = json.loads(sample["ground_truth"])

    n_valid = sum(len(l["words"]) for l in gt["valid_line"])
    print(f"valid_line words: {n_valid}")

    for key in ("dontcare", "roi", "repeating_symbol"):
        val = gt.get(key)
        print(f"\n--- {key} ---")
        print("type:", type(val).__name__, "len:" , len(val) if hasattr(val, "__len__") else "n/a")
        print(json.dumps(val, ensure_ascii=False)[:600])

    print("\n--- gt_parse ---")
    print(json.dumps(gt["gt_parse"], ensure_ascii=False)[:900])

    # how many docs have non-empty dontcare
    n = sum(1 for s in ds["train"].select(range(100))
            if json.loads(s["ground_truth"]).get("dontcare"))
    print(f"\ndocs (of first 100) with non-empty dontcare: {n}")


if __name__ == "__main__":
    main()