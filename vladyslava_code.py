import re
import sqlite3
import argparse
from pathlib import Path

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

try:
    from nltk.tokenize import wordpunct_tokenize
except ImportError:
    wordpunct_tokenize = None

try:
    from scipy import stats
except ImportError:
    stats = None


DATA_DIR = Path("vladyslava_data")
FIGURE_DIR = DATA_DIR / "figures"
TIME_ORDER = ["morning", "afternoon", "evening", "night"]
MIN_WORDS_FOR_ANALYSIS = 5


def ensure_dirs():
    DATA_DIR.mkdir(exist_ok=True)
    FIGURE_DIR.mkdir(exist_ok=True)


def print_step(text):
    print(f"\n--- {text} ---")


def save_csv(df, path):
    df.to_csv(path, index=False)
    print(f"Saved: {path}")


def read_dataset_csv(input_csv):
    attempts = [
        {"encoding": "utf-8"},
        {"encoding": "latin1"},
        {"encoding": "cp1252"},
        {"encoding": "utf-8", "engine": "python", "sep": None, "on_bad_lines": "skip"},
        {"encoding": "latin1", "engine": "python", "sep": None, "on_bad_lines": "skip"},
        {"encoding": "cp1252", "engine": "python", "sep": None, "on_bad_lines": "skip"}
    ]

    last_error = None

    for options in attempts:
        try:
            print(f"Trying to read CSV with options: {options}")
            df = pd.read_csv(input_csv, **options)
            print("CSV loaded successfully.")
            return df
        except (UnicodeDecodeError, pd.errors.ParserError) as error:
            last_error = error
            print(f"Reading attempt failed: {error}")

    raise ValueError("Could not read the CSV file.") from last_error


def clean_column_name(name):
    name = str(name).lower().strip()
    name = name.replace(";", "")
    name = re.sub(r"[^a-z0-9_]", "", name)
    return name


def find_col(df, names):
    mapping = {clean_column_name(c): c for c in df.columns}

    for name in names:
        clean_name = clean_column_name(name)
        if clean_name in mapping:
            return mapping[clean_name]

    return None


def build_text(df):
    text_col = find_col(
        df,
        ["text", "body", "comment", "content", "selftext", "post_text", "description"]
    )
    title_col = find_col(df, ["title", "post_title"])

    if text_col and title_col:
        return (
            df[title_col].fillna("").astype(str)
            + " "
            + df[text_col].fillna("").astype(str)
        ).str.strip()

    if text_col:
        return df[text_col].fillna("").astype(str)

    if title_col:
        return df[title_col].fillna("").astype(str)

    raise ValueError("No text column found.")


def build_time(df):
    time_col = find_col(
        df,
        [
            "created_utc",
            "timestamp",
            "date",
            "datetime",
            "created_datetime",
            "created_at",
            "time",
            "created"
        ]
    )

    if not time_col:
        raise ValueError("No time column found.")

    raw = df[time_col].astype(str).str.replace(";", "", regex=False).str.strip()
    numeric = pd.to_numeric(raw, errors="coerce")

    if numeric.notna().mean() > 0.7:
        median_value = numeric.dropna().median()

        if median_value > 10**17:
            created_utc = numeric / 10**9
        elif median_value > 10**14:
            created_utc = numeric / 10**6
        elif median_value > 10**12:
            created_utc = numeric / 1000
        else:
            created_utc = numeric

        return created_utc.astype("float")

    dt = pd.to_datetime(raw, utc=True, errors="coerce")

    if dt.notna().sum() == 0:
        raise ValueError(f"Could not parse time column: {time_col}")

    created_utc = (dt.astype("int64") // 10**9).astype("float")
    return created_utc


def normalize_dataset(input_csv):
    print_step(f"Loading dataset: {input_csv}")

    df = read_dataset_csv(input_csv)

    df.columns = [str(c).strip() for c in df.columns]
    print("Columns found:", list(df.columns))

    text = build_text(df)
    created_utc = build_time(df)

    id_col = find_col(df, ["id", "record_id", "comment_id", "post_id"])
    score_col = find_col(df, ["score", "ups", "upvotes"])
    subreddit_col = find_col(df, ["subreddit", "subreddit_name"])
    permalink_col = find_col(df, ["permalink", "url", "link"])
    comments_col = find_col(df, ["num_comments", "comments"])
    time_check_col = find_col(df, ["time_of_day_check"]) 

    out = pd.DataFrame({
        "record_id": (
            df[id_col].astype(str)
            if id_col
            else [f"local_{i}" for i in range(len(df))]
        ),
        "record_type": "dataset_text",
        "subreddit": (
            df[subreddit_col].astype(str)
            if subreddit_col
            else "unknown"
        ),
        "title": "",
        "text": text,
        "score": (
            pd.to_numeric(df[score_col], errors="coerce")
            if score_col
            else np.nan
        ),
        "num_comments": (
            pd.to_numeric(df[comments_col], errors="coerce")
            if comments_col
            else np.nan
        ),
        "created_utc": created_utc,
                "time_of_day_check": (
            df[time_check_col].astype(str)
            if time_check_col
            else ""
        ),
        "permalink": (
            df[permalink_col].astype(str)
            if permalink_col
            else ""
        )
    })

    out = out.dropna(subset=["created_utc"])
    out = out[out["text"].astype(str).str.strip().ne("")].copy()

    save_csv(out, DATA_DIR / "reddit_raw.csv")
    return out


def clean_text(text):
    if pd.isna(text):
        return ""

    text = str(text).lower()

    text = re.sub(r"http\S+|www\.\S+", " ", text)
    text = re.sub(r"u/[A-Za-z0-9_-]+", " ", text)
    text = re.sub(r"r/[A-Za-z0-9_-]+", " ", text)
    text = text.replace("[deleted]", " ").replace("[removed]", " ")
    text = re.sub(r"[^a-zA-Z\s']", " ", text)
    text = re.sub(r"\s+", " ", text).strip()

    return text


def tokenize(text):
    if not text:
        return []

    if wordpunct_tokenize:
        return [
            t.lower()
            for t in wordpunct_tokenize(text)
            if re.search(r"[a-zA-Z]", t)
        ]

    return re.findall(r"[a-zA-Z']+", text.lower())


def time_of_day(hour):
    if 5 <= hour <= 11:
        return "morning"

    if 12 <= hour <= 16:
        return "afternoon"

    if 17 <= hour <= 20:
        return "evening"

    return "night"


def preprocess(df):
    print_step("Cleaning and preprocessing")

    df = df.copy()

    df["clean_text"] = df["text"].apply(clean_text)
    df["tokens"] = df["clean_text"].apply(tokenize)
    df["word_count"] = df["tokens"].apply(len)

    df["created_datetime_utc"] = pd.to_datetime(
        df["created_utc"],
        unit="s",
        utc=True,
        errors="coerce"
    )

    df = df.dropna(subset=["created_datetime_utc"])

    df["hour_utc"] = df["created_datetime_utc"].dt.hour

    if "time_of_day_check" in df.columns and df["time_of_day_check"].astype(str).str.strip().ne("").any():
        df["time_of_day"] = df["time_of_day_check"].astype(str).str.strip()
    else:
        df["time_of_day"] = df["hour_utc"].apply(time_of_day)

    df = df[df["word_count"] >= MIN_WORDS_FOR_ANALYSIS].copy()

    save_csv(
        df.drop(columns=["tokens"], errors="ignore"),
        DATA_DIR / "reddit_processed.csv"
    )

    return df


def load_vad():
    print_step("Loading NRC VAD Lexicon")

    path = DATA_DIR / "NRC-VAD-Lexicon.txt"

    if not path.exists():
        raise FileNotFoundError("Put NRC-VAD-Lexicon.txt inside vladyslava_data/")

    rows = []

    with open(path, "r", encoding="utf-8", errors="ignore") as file:
        for line in file:
            line = line.strip()

            if not line:
                continue

            parts = line.split()

            if len(parts) < 4:
                continue

            term = " ".join(parts[:-3]).lower().strip()
            valence = parts[-3]
            arousal = parts[-2]
            dominance = parts[-1]

            try:
                rows.append({
                    "term": term,
                    "valence": float(valence),
                    "arousal": float(arousal),
                    "dominance": float(dominance)
                })
            except ValueError:
                continue

    vad = pd.DataFrame(rows)

    vad = vad.dropna(
        subset=["term", "valence", "arousal"]
    ).drop_duplicates("term")

    print(f"Loaded terms: {len(vad)}")

    return vad.set_index("term").to_dict("index")


def score_one(tokens, vad):
    vals = []
    aros = []

    for tok in tokens:
        if tok in vad:
            vals.append(vad[tok]["valence"])
            aros.append(vad[tok]["arousal"])

    if not vals:
        return pd.Series({
            "valence": np.nan,
            "arousal": np.nan,
            "vad_word_count": 0,
            "vad_coverage": 0.0
        })

    return pd.Series({
        "valence": float(np.mean(vals)),
        "arousal": float(np.mean(aros)),
        "vad_word_count": len(vals),
        "vad_coverage": len(vals) / max(len(tokens), 1)
    })


def score_all(df, vad):
    print_step("Scoring VAD")

    scores = df["tokens"].apply(lambda x: score_one(x, vad))

    df = pd.concat([df, scores], axis=1)
    df = df.dropna(subset=["valence", "arousal"])

    save_csv(
        df.drop(columns=["tokens"], errors="ignore"),
        DATA_DIR / "vad_results.csv"
    )

    return df


def summarize(df):
    print_step("Creating summaries")

    summary = df.groupby("time_of_day").agg(
        n_texts=("record_id", "count"),
        mean_valence=("valence", "mean"),
        sd_valence=("valence", "std"),
        mean_arousal=("arousal", "mean"),
        sd_arousal=("arousal", "std"),
        mean_word_count=("word_count", "mean"),
        mean_vad_coverage=("vad_coverage", "mean")
    ).reset_index()

    summary["time_of_day"] = pd.Categorical(
        summary["time_of_day"],
        TIME_ORDER,
        ordered=True
    )

    summary = summary.sort_values("time_of_day")

    save_csv(summary, DATA_DIR / "group_summary.csv")

    hourly = df.groupby("hour_utc").agg(
        n_texts=("record_id", "count"),
        mean_valence=("valence", "mean"),
        mean_arousal=("arousal", "mean")
    ).reset_index().sort_values("hour_utc")

    save_csv(hourly, DATA_DIR / "hourly_summary.csv")

    print(summary)

    return summary, hourly


def statistical_tests(df):
    print_step("Running statistical tests")

    if stats is None:
        print("SciPy not installed; skipping statistics.")
        return pd.DataFrame()

    rows = []

    for metric in ["valence", "arousal"]:
        groups = [
            g[metric].dropna().values
            for _, g in df.groupby("time_of_day")
            if len(g[metric].dropna()) > 1
        ]

        if len(groups) >= 2:
            anova = stats.f_oneway(*groups)
            kruskal = stats.kruskal(*groups)

            rows.append({
                "metric": metric,
                "test": "One-way ANOVA",
                "statistic": anova.statistic,
                "p_value": anova.pvalue
            })

            rows.append({
                "metric": metric,
                "test": "Kruskal-Wallis",
                "statistic": kruskal.statistic,
                "p_value": kruskal.pvalue
            })

    out = pd.DataFrame(rows)

    save_csv(out, DATA_DIR / "statistical_tests.csv")

    return out


def savefig(path):
    plt.tight_layout()
    plt.savefig(path, dpi=300)
    plt.close()
    print(f"Saved figure: {path}")


def make_figures(df, summary, hourly):
    print_step("Creating visualizations")

    plt.figure(figsize=(8, 5))
    plt.bar(
        summary["time_of_day"].astype(str),
        summary["n_texts"],
        label="Number of texts"
    )
    plt.title("Balanced Reddit Text Sample by Time of Day")
    plt.xlabel("Time of Day")
    plt.ylabel("Number of Texts")
    plt.legend()
    savefig(FIGURE_DIR / "fig1_record_counts_by_time.png")

    plt.figure(figsize=(8, 5))
    plt.bar(
        summary["time_of_day"].astype(str),
        summary["mean_valence"],
        yerr=summary["sd_valence"],
        capsize=5,
        label="Mean valence"
    )
    plt.title("Mean Valence by Time of Day")
    plt.xlabel("Time of Day")
    plt.ylabel("Mean Valence")
    plt.legend()
    savefig(FIGURE_DIR / "fig2_mean_valence_by_time.png")

    plt.figure(figsize=(8, 5))
    plt.bar(
        summary["time_of_day"].astype(str),
        summary["mean_arousal"],
        yerr=summary["sd_arousal"],
        capsize=5,
        label="Mean arousal"
    )
    plt.title("Mean Arousal by Time of Day")
    plt.xlabel("Time of Day")
    plt.ylabel("Mean Arousal")
    plt.legend()
    savefig(FIGURE_DIR / "fig3_mean_arousal_by_time.png")

    summary_plot = summary.copy()
    summary_plot["time_of_day"] = pd.Categorical(
        summary_plot["time_of_day"],
        TIME_ORDER,
        ordered=True
    )
    summary_plot = summary_plot.sort_values("time_of_day")

    plt.figure(figsize=(9, 5))
    plt.plot(
        summary_plot["time_of_day"].astype(str),
        summary_plot["mean_valence"],
        marker="o",
        label="Valence"
    )
    plt.plot(
        summary_plot["time_of_day"].astype(str),
        summary_plot["mean_arousal"],
        marker="o",
        label="Arousal"
    )
    plt.title("Mean Emotional Tone by Time of Day")
    plt.xlabel("Time of Day")
    plt.ylabel("Mean VAD Score")
    plt.legend()
    savefig(FIGURE_DIR / "fig4_emotional_tone_by_time_group.png")

    sample = df.sample(min(len(df), 1000), random_state=42)

    plt.figure(figsize=(7, 5))

    for name, part in sample.groupby("time_of_day"):
        plt.scatter(
            part["valence"],
            part["arousal"],
            alpha=0.55,
            label=str(name)
        )

    plt.title("Valence and Arousal Distribution")
    plt.xlabel("Valence")
    plt.ylabel("Arousal")
    plt.legend()
    savefig(FIGURE_DIR / "fig5_valence_arousal_scatter.png")

    captions = DATA_DIR / "figure_captions.txt"

    captions.write_text(
    "Figure 1: Balanced sample of Reddit relationship_advice texts across four time-of-day groups.\n"
    "Figure 2: Mean valence by time of day. Lower values indicate more negative emotional tone.\n"
    "Figure 3: Mean arousal by time of day. Higher values indicate stronger emotional intensity.\n"
    "Figure 4: Mean emotional tone by time-of-day group, comparing valence and arousal.\n"
    "Figure 5: Scatter plot of valence and arousal scores by time-of-day group.\n",
    encoding="utf-8"
)

    print(f"Saved captions: {captions}")


def save_sqlite(raw, processed, results, summary, hourly, tests):
    db = DATA_DIR / "reddit_vad_project.sqlite"

    con = sqlite3.connect(db)

    raw.to_sql(
        "raw_reddit_data",
        con,
        if_exists="replace",
        index=False
    )

    processed.drop(
        columns=["tokens"],
        errors="ignore"
    ).to_sql(
        "processed_reddit_data",
        con,
        if_exists="replace",
        index=False
    )

    results.drop(
        columns=["tokens"],
        errors="ignore"
    ).to_sql(
        "vad_results",
        con,
        if_exists="replace",
        index=False
    )

    summary.to_sql(
        "group_summary",
        con,
        if_exists="replace",
        index=False
    )

    hourly.to_sql(
        "hourly_summary",
        con,
        if_exists="replace",
        index=False
    )

    if not tests.empty:
        tests.to_sql(
            "statistical_tests",
            con,
            if_exists="replace",
            index=False
        )

    con.close()

    print(f"Saved SQLite database: {db}")


def notes(summary, tests):
    path = DATA_DIR / "interpretation_notes.txt"

    low_v = summary.loc[summary["mean_valence"].idxmin()]
    high_a = summary.loc[summary["mean_arousal"].idxmax()]

    path.write_text(
        f"Lowest mean valence: {low_v['time_of_day']} "
        f"({low_v['mean_valence']:.3f}).\n"
        f"Highest mean arousal: {high_a['time_of_day']} "
        f"({high_a['mean_arousal']:.3f}).\n"
        "Compare this with H1: lower valence at night, and H2: higher arousal in evening/night.\n"
        "Limitation: timestamps are UTC and may not represent each Reddit user's local time.\n",
        encoding="utf-8"
    )

    print(f"Saved notes: {path}")


def main():
    parser = argparse.ArgumentParser(
        description="Dataset-based Reddit VAD analysis; no Reddit API required."
    )

    parser.add_argument(
        "--input-csv",
        required=True,
        help="Path to downloaded Reddit CSV dataset."
    )

    args = parser.parse_args()

    ensure_dirs()

    print_step("Temporal Variations in Emotional Tone in Reddit Discussions")
    print("Running dataset version. No Reddit API credentials are required.")

    raw = normalize_dataset(args.input_csv)
    processed = preprocess(raw)
    vad = load_vad()
    results = score_all(processed, vad)
    summary, hourly = summarize(results)
    tests = statistical_tests(results)

    make_figures(results, summary, hourly)
    save_sqlite(raw, processed, results, summary, hourly, tests)
    notes(summary, tests)

    print_step("Pipeline completed successfully")
    print(f"All outputs are inside: {DATA_DIR}")


if __name__ == "__main__":
    main()