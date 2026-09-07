"""Smoke check: can we load CORD and read one sample?"""
import json
from datasets import load_dataset


def main() -> None:
    ds = load_dataset("naver-clova-ix/cord-v2")
    print("splits:", {k: len(v) for k, v in ds.items()})

    sample = ds["train"][0]
    print("image size:", sample["image"].size)

    gt = json.loads(sample["ground_truth"])
    print("gt top-level keys:", list(gt.keys()))

    lines = gt["valid_line"]
    print(f"\n{len(lines)} line groups. First one:")
    print(json.dumps(lines[0], indent=2, ensure_ascii=False)[:800])


if __name__ == "__main__":
    main()