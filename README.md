# ReviewSense - Review Sentiment Analyzer

MSc coursework project - a small web app that reads a customer review and predicts whether it's positive or negative, using a TF-IDF + Logistic Regression model. There's a simple web page to try it out, plus a REST API endpoint for the same thing.

Live demo: https://chethana.pythonanywhere.com

## Why this exists

Companies get a lot of review/feedback text - product reviews, app reviews, support tickets, etc. Reading through all of it by hand doesn't scale, so an automated way to flag whether feedback is positive or negative is genuinely useful (even just as a quick triage step before a human reads it).

For this assignment the goal was to build the full pipeline end to end - dataset, model training, a working web app, an API, and deployment - rather than just training a model in a notebook and stopping there.

Who'd actually use something like this:
- A small e-commerce/SaaS team wanting to auto-tag reviews as they come in
- A PM who wants a rough sentiment trend over time without reading every review
- Another developer who just wants a sentiment endpoint to call from a bigger pipeline

## How it works

Nothing fancy on purpose. I went with a classic ML approach instead of a transformer model because:

- it trains in a couple of seconds on a normal laptop, no GPU
- the saved model is tiny (a few KB), which matters because I'm deploying on a free-tier host
- it's easy to explain in a viva/demo - I can actually point at which words pushed the prediction one way or the other, which isn't really true for a black-box deep model

Pipeline is:
1. User types/pastes a review into the web form (or sends it to `/api/predict`)
2. Text gets vectorized with TF-IDF (unigrams + bigrams, stopwords removed)
3. A Logistic Regression classifier predicts positive/negative + a confidence score
4. Result comes back as JSON and gets shown on the page

## Dataset

I didn't use a downloaded dataset for this - `data/generate_dataset.py` generates one programmatically instead. It combines 15 subject categories (product, movie, restaurant, phone, laptop, etc.) with ~20 positive and ~20 negative sentence templates, so you get a decent variety of review-like sentences (~240 rows total) without needing to download or license anything.

Honest caveat: because the data comes from templates, the model gets basically 100% accuracy on the held-out test split. That's not a real-world accuracy number - it's just the model correctly recognizing the same phrasing patterns it was trained on. I'm keeping this in the README instead of hiding it, since a real dataset would obviously give a more modest (and more meaningful) accuracy figure, probably somewhere in the 85-92% range for a TF-IDF + LogReg setup on review text.

If I (or someone else) wanted to swap in a real dataset, e.g. [IMDB reviews](https://ai.stanford.edu/~amaas/data/sentiment/) or the [Amazon Polarity dataset](https://huggingface.co/datasets/amazon_polarity), you'd just need to replace `data/reviews.csv` (same `text,label` columns) and rerun `model/train.py`. Nothing else in the app needs to change.

## Model / tooling used

- Feature extraction: `TfidfVectorizer` (unigrams + bigrams, English stopwords removed)
- Classifier: `LogisticRegression` from scikit-learn
- Train/test split: 80/20, stratified
- Saved with `joblib` so the Flask app can just load it at startup instead of retraining every time

## Project structure

```
review-sentiment-ai/
├── app/
│   ├── __init__.py
│   └── app.py                    # Flask app - serves the UI + API
├── data/
│   ├── generate_dataset.py       # builds the training data
│   └── reviews.csv               # generated dataset
├── model/
│   ├── train.py                  # trains + saves the model
│   └── sentiment_pipeline.joblib # trained model
├── templates/
│   └── index.html                # web UI
├── static/
├── Dockerfile
├── requirements.txt
└── render.yaml
```

`app/app.py` handles three routes:
- `GET /` - the web page
- `POST /api/predict` - takes `{"text": "..."}`, returns the predicted label + confidence
- `GET /api/health` - basic health check (mostly useful for whatever platform is hosting it)

## Running it locally

```bash
git clone https://github.com/chethuuu/review-sentiment-ai.git
cd review-sentiment-ai

python3 -m venv venv
source venv/bin/activate        # on Windows: venv\Scripts\activate

pip install -r requirements.txt

# these two are already generated/trained and committed,
# but you can regenerate them from scratch if you want
python3 data/generate_dataset.py
python3 model/train.py

python3 app/app.py
```

App runs on http://localhost:8080.

## Deployment

Deployed on **PythonAnywhere** (free tier). I went with this over Render/Cloud Run mainly because it doesn't need a card to sign up and it runs Flask apps natively, so I didn't even need the Dockerfile for this particular deployment - PythonAnywhere just runs the app through their WSGI setup.

Live app: https://chethana.pythonanywhere.com

Steps I followed:
1. Signed up for a free PythonAnywhere account (Beginner plan, no card needed)
2. Opened a Bash console there and cloned the repo:
   ```bash
   git clone https://github.com/chethuuu/review-sentiment-ai.git
   cd review-sentiment-ai
   pip3.13 install --user -r requirements.txt
   ```
3. Web tab → Add a new web app → Manual configuration → picked the matching Python version
4. Set the source/working directory to `/home/<username>/review-sentiment-ai`
5. Edited the WSGI file it gives you so it points at the Flask app:
   ```python
   import sys
   project_home = '/home/<username>/review-sentiment-ai'
   if project_home not in sys.path:
       sys.path.insert(0, project_home)
   from app.app import app as application
   ```
6. Hit Reload on the Web tab and it was live at `https://<username>.pythonanywhere.com`

I kept the `Dockerfile` and `render.yaml` in the repo anyway since I built them first before switching to PythonAnywhere, and they still work if someone wants to deploy this as a container instead (Render, Cloud Run, App Runner, Azure App Service - anywhere that takes a Docker image on port 8080). Didn't want to just delete that work.

## Using the API

Health check:
```bash
curl https://chethana.pythonanywhere.com/api/health
# {"status": "ok"}
```

Predict:
```bash
curl -X POST https://chethana.pythonanywhere.com/api/predict \
  -H "Content-Type: application/json" \
  -d '{"text": "This product completely exceeded my expectations!"}'
```

Response:
```json
{
  "text": "This product completely exceeded my expectations!",
  "label": "positive",
  "confidence": 0.87
}
```

If you send empty/missing text you get a 400 back:
```json
{ "error": "Field 'text' is required and cannot be empty." }
```

## Running with Docker (optional)

```bash
docker build -t reviewsense .
docker run -p 8080:8080 reviewsense
```

## Retraining after changing the dataset

```bash
python3 data/generate_dataset.py
python3 model/train.py
```

---

Built for an MSc assignment - not intended as a production-grade sentiment tool, more a demonstration of putting a full ML pipeline (data → model → API → deployed app) together end to end.
