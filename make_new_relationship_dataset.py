from datetime import datetime, timezone
from pathlib import Path

import pandas as pd
from datasets import load_dataset


DATA_DIR = Path("vladyslava_data")
DATA_DIR.mkdir(exist_ok=True)

OUT_FILE = DATA_DIR / "relationship_advice_4groups.csv"

TARGET_PER_GROUP = 500

TIME_ORDER = ["morning", "afternoon", "evening", "night"]


def time_of_day(hour):
    if 5 <= hour <= 11:
        return "morning"

    if 12 <= hour <= 16:
        return "afternoon"

    if 17 <= hour <= 20:
        return "evening"

    return "night"


def clean_value(value):
    if value is None:
        return ""

    value = str(value).strip()

    if value.lower() in ["none", "null", "nan", "[deleted]", "[removed]"]:
        return ""

    return value


def load_relationship_dataset():
    return load_dataset(
        "HuggingFaceGECLM/REDDIT_submissions",
        split="relationship_advice",
        streaming=True
    )


def main():
    dataset = load_relationship_dataset()

    counts = {group: 0 for group in TIME_ORDER}
    rows = []
    checked = 0

    for item in dataset:
        checked += 1

        created_raw = clean_value(item.get("created_utc"))

        try:
            created_utc = float(created_raw)
        except ValueError:
            continue

        hour = datetime.fromtimestamp(created_utc, tz=timezone.utc).hour
        group = time_of_day(hour)

        if counts[group] >= TARGET_PER_GROUP:
            continue

        title = clean_value(item.get("title"))
        selftext = clean_value(item.get("selftext"))

        full_text = f"{title} {selftext}".strip()

        if len(full_text.split()) < 10:
            continue

        rows.append({
            "id": clean_value(item.get("id")),
            "title": title,
            "selftext": selftext,
            "text": full_text,
            "created_utc": int(created_utc),
            "time_of_day_check": group,
            "score": clean_value(item.get("score")),
            "num_comments": clean_value(item.get("num_comments")),
            "subreddit": clean_value(item.get("subreddit")),
            "permalink": clean_value(item.get("permalink")),
            "url": clean_value(item.get("url"))
        })

        counts[group] += 1

        print(counts)

        if all(counts[group] >= TARGET_PER_GROUP for group in TIME_ORDER):
            break

    df = pd.DataFrame(rows)
    df.to_csv(OUT_FILE, index=False, encoding="utf-8")

    print()
    print("Done.")
    print(f"Checked records: {checked}")
    print(f"Saved records: {len(df)}")
    print(f"Saved file: {OUT_FILE}")
    print(counts)


if __name__ == "__main__":
    main()