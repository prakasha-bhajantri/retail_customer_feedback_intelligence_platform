## Deployment

This project is deployed on Google Cloud Platform using:

- Google Cloud Run
- Google Cloud Build
- Artifact Registry
- Google Cloud Storage

Deployment flow:

GitHub
    ↓
Cloud Build
    ↓
Artifact Registry
    ↓
Cloud Run

The application automatically downloads trained models from Google Cloud Storage during startup if they are not available locally.


## Model Management

To keep Docker images lightweight, trained models are not packaged inside the container.

Models are stored in a Google Cloud Storage bucket.

Bucket:

gs://retail-feedback-models

Current models:

- Sentiment Analysis Model
- Retail Named Entity Recognition (NER) Model

Application startup flow:

1. Check if models already exist locally.
2. If present, load immediately.
3. Otherwise, download from Google Cloud Storage.
4. Load models into memory.


## Cloud Storage Structure

```text
gs://retail-feedback-models/

sentiment/
    best_model/
        config.json
        tokenizer.json
        tokenizer_config.json
        model.safetensors

retail_ner/
    best_model/
        config.json
        tokenizer.json
        tokenizer_config.json
        model.safetensors
```

## Local Development

Install dependencies

```bash
pip install -r requirements.txt
```

Authenticate once

```bash
gcloud auth application-default login
```

Run

```bash
streamlit run streamlit_app/app.py
```

If models are missing locally, they are downloaded automatically from Google Cloud Storage.


## Docker

Build

```bash
docker build -f deployment/Dockerfile -t retail-feedback:latest .
```

Run

```bash
docker run \
-p 8501:8501 \
-v ~/.config/gcloud:/home/appuser/.config/gcloud:ro \
-e GOOGLE_APPLICATION_CREDENTIALS=/home/appuser/.config/gcloud/application_default_credentials.json \
retail-feedback:latest
```


## Cloud Run

Deploy

```bash
./deployment/scripts/deploy_cloud_run.sh
```

Cloud Run uses its Service Account to access Google Cloud Storage.

No credentials are required inside the container.