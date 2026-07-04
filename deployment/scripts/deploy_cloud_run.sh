#!/bin/bash

set -e

PROJECT_ID=$(gcloud config get-value project)
REGION="asia-south1"
SERVICE_NAME="retail-feedback"
REPOSITORY="retail-feedback"
IMAGE="retail-feedback-intelligence"

echo "========================================"
echo "Deploying Retail Feedback Platform"
echo "========================================"
echo "Project   : ${PROJECT_ID}"
echo "Region    : ${REGION}"
echo "Service   : ${SERVICE_NAME}"
echo ""

gcloud run deploy ${SERVICE_NAME} \
    --image ${REGION}-docker.pkg.dev/${PROJECT_ID}/${REPOSITORY}/${IMAGE}:latest \
    --platform managed \
    --region ${REGION} \
    --allow-unauthenticated \
    --port 8501 \
    --memory 4Gi \
    --cpu 2 \
    --timeout 300 \
    --min-instances 0 \
    --max-instances 2 \
    --set-env-vars APP_ENV=prod

echo ""
echo "========================================"
echo "Deployment Complete"
echo "========================================"