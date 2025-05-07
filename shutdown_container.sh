export PROJECT_ID=soe-iris-gcp
export APP=leaderboard 
export PORT=10000
export REGION="us-central1"
export TAG="gcr.io/$PROJECT_ID/$APP"

gcloud run services delete $APP --region $REGION

# Check if it disAPPeared (optional)
gcloud run services list