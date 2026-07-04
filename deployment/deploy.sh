#!/bin/bash

gcloud run deploy retail-feedback \
    --image asia-south1-docker.pkg.dev/$GOOGLE_CLOUD_PROJECT/retail-feedback/retail-feedback-intelligence:latest \
    --platform managed \
    --region asia-south1 \
    --allow-unauthenticated \
    --port 8501 \
    --memory 4Gi \
    --cpu 2 \
    --timeout 300 \
    --min-instances 0 \
    --max-instances 2