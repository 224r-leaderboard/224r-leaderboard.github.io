export PROJECT_ID=soe-iris-gcp
export APP=leaderboard 
export PORT=10000
export REGION="us-central1"
export TAG="gcr.io/$PROJECT_ID/$APP"

# Set Default Project (all later commands will use it) 
gcloud config set project $PROJECT_ID

docker build -t $TAG .
docker run -dp $PORT:$PORT -e PORT=$PORT $TAG

gcloud builds submit --tag $TAG
gcloud run deploy $APP --image $TAG --platform managed --region $REGION --allow-unauthenticated