"""Profile CORD categories to inform the label schema."""
import json
from collections import Counter, defaultdict
from datasets import load_dataset


def main() -> None:
    ds = load_dataset("naver-clova-ix/cord-v2")

    group_counts = Counter()
    word_counts = Counter()
    doc_counts = Counter()
    words_per_group = []
    groups_per_doc = []

    for sample in ds["train"]:
        gt = json.loads(sample["ground_truth"])
        seen = set()
        n_groups = 0
        for line in gt["valid_line"]:
            cat = line["category"]
            group_counts[cat] += 1
            word_counts[cat] += len(line["words"])
            words_per_group.append(len(line["words"]))
            seen.add(cat)
            n_groups += 1
        groups_per_doc.append(n_groups)
        for cat in seen:
            doc_counts[cat] += 1

    print(f"{'category':<34}{'groups':>8}{'words':>8}{'docs':>7}")
    print("-" * 57)
    for cat, n in group_counts.most_common():
        print(f"{cat:<34}{n:>8}{word_counts[cat]:>8}{doc_counts[cat]:>7}")

    n = len(words_per_group)
    print(f"\ncategories: {len(group_counts)}   train docs: {len(ds['train'])}")
    print(f"words/group: mean {sum(words_per_group)/n:.1f}, max {max(words_per_group)}")
    print(f"groups/doc:  mean {sum(groups_per_doc)/len(groups_per_doc):.1f}, "
          f"max {max(groups_per_doc)}")

    # How group_id partitions one document — the line-item question
    gt = json.loads(ds["train"][0]["ground_truth"])
    buckets = defaultdict(list)
    for line in gt["valid_line"]:
        buckets[(line["group_id"], line["sub_group_id"])].append(line["category"])

    print("\nfirst doc — (group_id, sub_group_id) -> categories:")
    for key in sorted(buckets)[:12]:
        print(f"  {key}: {buckets[key]}")


if __name__ == "__main__":
    main()