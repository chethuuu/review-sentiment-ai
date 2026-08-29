"""
Builds data/reviews.csv from the NLTK movie_reviews corpus - 2,000 real,
human-written movie reviews (1000 positive / 1000 negative), originally
collected by Pang & Lee for their polarity classification research and
distributed publicly through NLTK. Downloads the corpus on first run.

Needs nltk installed (not a runtime dependency of the app itself):
    pip install nltk
"""
import csv

import nltk

nltk.download("movie_reviews", quiet=True)
from nltk.corpus import movie_reviews  # noqa: E402

OUT_PATH = "data/reviews.csv"


def main():
    rows = []
    for category, label in (("pos", "positive"), ("neg", "negative")):
        for fileid in movie_reviews.fileids(category):
            text = movie_reviews.raw(fileid).replace("\n", " ").strip()
            rows.append((text, label))

    with open(OUT_PATH, "w", newline="", encoding="utf-8") as f:
        writer = csv.writer(f)
        writer.writerow(["text", "label"])
        writer.writerows(rows)

    print(f"Wrote {len(rows)} reviews to {OUT_PATH}")


if __name__ == "__main__":
    main()
