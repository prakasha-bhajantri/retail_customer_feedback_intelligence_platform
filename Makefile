run:
	streamlit run streamlit_app/app.py

docker-build:
	docker build -f deployment/Dockerfile -t retail-feedback:latest .

docker-run:
	docker run -p 8501:8501 retail-feedback:latest

cloud-build:
	gcloud builds submit --config deployment/cloudbuild.yaml

cloud-deploy:
	gcloud run deploy retail-feedback