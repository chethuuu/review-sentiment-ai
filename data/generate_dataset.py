"""
Generates a labeled dataset of product/movie/service reviews for sentiment
analysis training. Template + vocabulary based, producing diverse,
non-repetitive sentence structures at scale (no external download needed).
"""
import csv
import random

random.seed(42)

subjects = ["product", "movie", "restaurant", "book", "phone", "laptop", "service",
            "app", "hotel", "game", "album", "show", "gadget", "course", "headphones"]

positive_templates = [
    "This {s} exceeded all my expectations, I'm thoroughly impressed.",
    "Absolutely loved this {s}, would recommend to everyone.",
    "One of the best {s}s I've experienced in a long time.",
    "The {s} was fantastic from start to finish.",
    "I'm so happy with this {s}, it's worth every penny.",
    "Great {s}! Everything worked perfectly and exceeded expectations.",
    "This {s} is a masterpiece, truly outstanding quality.",
    "Superb {s}, I can't stop recommending it to friends.",
    "Wonderful experience with this {s}, five stars all the way.",
    "The {s} delivered on every promise, brilliant execution.",
    "Impressive {s}, the attention to detail is remarkable.",
    "I am delighted with this {s}, it made my day.",
    "Top notch {s}, customer support was also excellent.",
    "This {s} is amazing, I've never been happier.",
    "Highly satisfied with the {s}, everything felt premium.",
    "Excellent {s}! Smooth, reliable, and beautifully designed.",
    "The {s} was a pleasant surprise, better than I hoped.",
    "Fantastic value for money, this {s} is a gem.",
    "I really enjoyed this {s}, it was engaging and well made.",
    "Brilliant {s}, exceeded my expectations in every way.",
]

negative_templates = [
    "This {s} was a complete disappointment, I regret buying it.",
    "Terrible {s}, nothing worked as advertised.",
    "One of the worst {s}s I've ever experienced.",
    "The {s} was awful from start to finish.",
    "I'm so unhappy with this {s}, total waste of money.",
    "Poor quality {s}, broke down almost immediately.",
    "This {s} is a mess, badly designed and unreliable.",
    "Horrible experience with this {s}, would not recommend.",
    "Disappointing {s}, customer support was unhelpful too.",
    "The {s} failed to deliver on its promises, frustrating.",
    "Mediocre {s} at best, definitely overpriced for what it offers.",
    "I am frustrated with this {s}, it ruined my day.",
    "Bad {s}, slow, buggy, and poorly built.",
    "This {s} is awful, I've never been so let down.",
    "Very unsatisfied with the {s}, everything felt cheap.",
    "Annoying {s}! Constant issues and unreliable performance.",
    "The {s} was an unpleasant surprise, worse than I feared.",
    "Poor value for money, this {s} is not worth it.",
    "I really disliked this {s}, it was boring and poorly made.",
    "Awful {s}, fell short of expectations in every way.",
]

def make_rows(templates, label, n_per_template=6):
    rows = []
    for t in templates:
        for _ in range(n_per_template):
            s = random.choice(subjects)
            text = t.format(s=s)
            rows.append((text, label))
    return rows

rows = []
rows += make_rows(positive_templates, "positive", n_per_template=6)
rows += make_rows(negative_templates, "negative", n_per_template=6)
random.shuffle(rows)

with open("data/reviews.csv", "w", newline="", encoding="utf-8") as f:
    writer = csv.writer(f)
    writer.writerow(["text", "label"])
    writer.writerows(rows)

print(f"Generated {len(rows)} labeled reviews -> data/reviews.csv")
