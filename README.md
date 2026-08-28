# ReviewSense — AI-Powered Review Sentiment Analyzer

An end-to-end NLP application that classifies customer/product reviews as **Positive**
or **Negative**, exposed through both a web interface and a REST API, containerized
with Docker and deployable to any major cloud platform.

---

## 1. Problem Statement

Businesses receive large volumes of unstructured text feedback every day — product
reviews, app store comments, support tickets, survey responses. Manually reading and
tagging this feedback to understand customer sentiment does not scale. Teams need a
fast, automated way to know, at a glance, whether feedback is trending positive or
negative so they can prioritize what to act on.

## 2. Use Case

ReviewSense can be used by:
- **E-commerce / SaaS teams** to automatically tag incoming reviews or support tickets
  by sentiment for triage and reporting.
- **Product managers** to monitor sentiment trends over time from exported review data.
- **Developers** as a drop-in sentiment API to embed into a larger pipeline (e.g. a
  dashboard that batches and visualizes review sentiment).
- **Students/researchers** as a lightweight, explainable baseline for text
  classification experiments.

A user (or another system) submits a piece of review text through the web form or the
API, and receives back a sentiment label plus a confidence score in real time.

## 3. Solution Overview

The application uses a classical, lightweight NLP pipeline rather than a large
pretrained model, by design — this keeps the model fast, cheap to run, easy to explain,
and simple to redeploy without any GPU or large model download at build time (which
matters when deploying on free-tier cloud infrastructure).

Pipeline:
1. Review text is submitted via the web UI or `POST /api/predict`.
2. The text is vectorized using **TF-IDF** (unigrams + bigrams).
3. A **Logistic Regression** classifier predicts the sentiment label and a
   probability-based confidence score.
4. The result is returned as JSON and rendered in the UI as a stamped verdict with a
   confidence meter.

## 4. Dataset

- **Source**: `data/generate_dataset.py` programmatically generates a labeled dataset
  of ~240 short reviews using a template + vocabulary approach across 15 subject
  categories (product, movie, restaurant, book, phone, laptop, service, app, hotel,
  game, album, show, gadget, course, headphones) with 20 distinct positive phrasing
  patterns and 20 distinct negative phrasing patterns.
- **Why generated data**: this keeps the project fully self-contained and
  reproducible with zero external downloads or licensing concerns, which is ideal for
  a demonstration/assignment context.
- **To extend with a real-world dataset**: swap `data/reviews.csv` for a public
  dataset such as the [IMDB Movie Reviews dataset](https://ai.stanford.edu/~amaas/data/sentiment/)
  or the [Amazon Product Reviews dataset](https://huggingface.co/datasets/amazon_polarity)
  (keeping the same `text,label` CSV schema), then rerun `model/train.py`. No other
  code changes are required.
- **Known limitation**: because the generated data follows a limited number of
  templates, the held-out test accuracy is artificially high (~100%). This is
  expected for template-based data and is called out here for transparency — it is
  not a claim of real-world accuracy. Retraining on a real dataset (see above) will
  give a more realistic accuracy figure, typically 85–92% for this type of
  TF-IDF + Logistic Regression pipeline on review sentiment tasks.

## 5. AI/ML Approach

| Component | Choice |
|---|---|
| Feature extraction | `TfidfVectorizer` (unigrams + bigrams, English stop words removed) |
| Model | `LogisticRegression` (scikit-learn) |
| Framework | scikit-learn 1.8 |
| Serialization | `joblib` |
| Evaluation | 80/20 stratified train/test split, accuracy + classification report |

This is a supervised binary text classification problem. TF-IDF + Logistic
Regression was chosen over a deep learning / transformer approach because it:
- Trains in seconds on CPU only, no GPU dependency
- Produces a tiny model artifact (~KBs), which keeps the Docker image small and cold
  starts fast on free-tier cloud platforms
- Is inherently interpretable (feature weights map directly to words/phrases)
- Is a well-established, appropriate baseline for this class of problem

## 6. Application Architecture

```
┌─────────────┐      HTTP       ┌──────────────────────┐
│   Browser   │ ───────────────>│   Flask Application    │
│  (Web UI)   │ <───────────────│   app/app.py           │
└─────────────┘     JSON/HTML   │                        │
                                 │  ┌──────────────────┐  │
┌─────────────┐      HTTP       │  │ sentiment_pipeline│  │
│  API client │ ───────────────>│  │   .joblib         │  │
│ (curl/etc.) │ <───────────────│  │ (TF-IDF + LogReg) │  │
└─────────────┘     JSON        │  └──────────────────┘  │
                                 └──────────────────────┘
                                          │
                                  Served via Gunicorn
                                  inside a Docker container
                                  deployed to the cloud
```

- `app/app.py` — Flask app: serves the UI (`GET /`), the prediction API
  (`POST /api/predict`), and a health check (`GET /api/health`) used by cloud
  platform probes.
- `templates/index.html` — self-contained HTML/CSS/JS front end (no build step).
- `model/train.py` — trains and serializes the ML pipeline.
- `model/sentiment_pipeline.joblib` — the trained, deployable model artifact.
- `data/generate_dataset.py` — generates the training dataset.

## 7. Technology Stack

- **Language**: Python 3.12
- **Web framework**: Flask 3.1
- **ML**: scikit-learn 1.8, joblib
- **Production WSGI server**: Gunicorn
- **Containerization**: Docker
- **Cloud deployment target**: Render.com (Docker-based web service) — instructions
  below also cover Railway, AWS App Runner/Elastic Beanstalk, Azure App Service, and
  Google Cloud Run, since the container is portable to any of them.
- **Version control**: Git / GitHub

## 8. Local Setup Instructions

```bash
# 1. Clone the repository
git clone <your-repo-url>
cd review-sentiment-ai

# 2. Create a virtual environment (recommended)
python3 -m venv venv
source venv/bin/activate      # Windows: venv\Scripts\activate

# 3. Install dependencies
pip install -r requirements.txt

# 4. Generate the dataset (already included, but regenerable)
python3 data/generate_dataset.py

# 5. Train the model (already included, but regenerable)
python3 model/train.py

# 6. Run the app locally
python3 app/app.py
# App will be available at http://localhost:8080
```

## 9. Deployment Details

The app is packaged as a single Docker container exposing port `8080` (configurable
via the `PORT` environment variable, which most cloud platforms inject
automatically). `render.yaml` is included as a ready-to-use blueprint for Render.

### Option A — Render.com (recommended, free tier available)
1. Push this repository to GitHub.
2. In the Render dashboard: **New → Blueprint**, connect your GitHub repo. Render
   will detect `render.yaml` and configure the service automatically.
   *(Or: New → Web Service → select the repo → Environment: Docker → it will pick
   up the `Dockerfile` automatically.)*
3. Render builds the Docker image and deploys it. Once live, your app is available
   at `https://<your-service-name>.onrender.com`.

### Option B — Google Cloud Run
```bash
gcloud builds submit --tag gcr.io/<PROJECT_ID>/reviewsense
gcloud run deploy reviewsense \
  --image gcr.io/<PROJECT_ID>/reviewsense \
  --platform managed \
  --port 8080 \
  --allow-unauthenticated
```

### Option C — AWS App Runner
1. Push the image to Amazon ECR:
   ```bash
   aws ecr create-repository --repository-name reviewsense
   docker build -t reviewsense .
   docker tag reviewsense:latest <account>.dkr.ecr.<region>.amazonaws.com/reviewsense:latest
   docker push <account>.dkr.ecr.<region>.amazonaws.com/reviewsense:latest
   ```
2. In the AWS App Runner console, create a service from that ECR image, set the port
   to `8080`, and deploy.

### Option D — Azure App Service (container)
```bash
az webapp create --resource-group <rg> --plan <plan> \
  --name reviewsense --deployment-container-image-name <your-registry>/reviewsense:latest
az webapp config appsettings set --resource-group <rg> --name reviewsense \
  --settings WEBSITES_PORT=8080
```

> **Note on submission**: Replace the placeholders above with your actual deployed
> URL once live, e.g.: `Live app: https://reviewsense.onrender.com`

## 10. API / Web Application Usage

### Web UI
Visit the deployed URL (or `http://localhost:8080` locally). Paste a review into the
text box and click **Stamp it** to see the predicted sentiment and confidence.

### REST API

**Health check**
```bash
curl https://<your-app-url>/api/health
# {"status": "ok"}
```

**Predict sentiment**
```bash
curl -X POST https://<your-app-url>/api/predict \
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

Error response (empty/missing text):
```json
{ "error": "Field 'text' is required and cannot be empty." }
```

## 11. Docker Instructions

Build and run the container locally:

```bash
# Build the image
docker build -t reviewsense .

# Run the container
docker run -p 8080:8080 reviewsense

# App available at http://localhost:8080
```

Push to Docker Hub (or any registry):
```bash
docker tag reviewsense <your-dockerhub-username>/reviewsense:latest
docker login
docker push <your-dockerhub-username>/reviewsense:latest
```

---

## Project Structure

```
review-sentiment-ai/
├── app/
│   ├── __init__.py
│   └── app.py                    # Flask application (UI + API)
├── data/
│   ├── generate_dataset.py       # Dataset generator
│   └── reviews.csv               # Generated training data
├── model/
│   ├── train.py                  # Training script
│   └── sentiment_pipeline.joblib # Trained model artifact
├── templates/
│   └── index.html                # Web UI
├── static/                       # (reserved for static assets)
├── Dockerfile
├── requirements.txt
├── render.yaml                   # Render.com deployment blueprint
├── .dockerignore
├── .gitignore
└── README.md
```

## Retraining the Model

To retrain (e.g. after editing the dataset or swapping in a real-world dataset):
```bash
python3 data/generate_dataset.py   # optional — regenerate/replace data/reviews.csv
python3 model/train.py             # retrains and overwrites the model artifact
```

## License

This project was created for academic/assignment purposes.
