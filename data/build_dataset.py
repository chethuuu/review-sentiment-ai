# downloads the NLTK movie_reviews corpus (2000 real reviews, Pang & Lee's
# polarity dataset) and writes it out as data/reviews.csv
# needs nltk installed - not needed to run the app itself, just this script:
#   pip install nltk
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
