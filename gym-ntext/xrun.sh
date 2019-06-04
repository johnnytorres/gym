#!/usr/bin/env bash


################################################
# USING STANDARD GOOGLE ML ENGINE:
################################################

#compile dependencies
python -m env.setup sdist

# REGION: select a region from https://cloud.google.com/ml-engine/docs/regions
# or use the default '`us-central1`'. The region is where the model will be deployed.
REGION="us-central1" #us-east1 #
TIER="BASIC_GPU" # BASIC | BASIC_GPU | STANDARD_1 | PREMIUM_1
PROJECT_ID="jt17383-research"
BUCKET="jt17383-research"
MODEL_NAME="deepq"
DATASET="imdb"
MODEL_DIR="gs://${BUCKET}/xdrl/${DATASET}_${MODEL_NAME}"
PACKAGE_PATH=agents # this can be a gcs location to a zipped and uploaded package
CURRENT_DATE=`date +%Y%m%d_%H%M%S`
# JOB_NAME: the name of your job running on Cloud ML Engine.
JOB_NAME=xdrl_${DATASET}_${MODEL_NAME}_${TIER}_${CURRENT_DATE}

echo "Model: ${JOB_NAME}"

gcloud ai-platform jobs submit training ${JOB_NAME} \
        --stream-logs \
        --job-dir=${MODEL_DIR} \
        --runtime-version=1.13 \
        --region=${REGION} \
        --module-name=agents.ntext_agent \
        --package-path=${PACKAGE_PATH}  \
        --config=config.yaml \
        -- \


# USING CUSTOM CONTAINER FOR GOOGLE ML ENGINE


## IMAGE_REPO_NAME: the image will be stored on Cloud Container Registry
#export IMAGE_REPO_NAME=xdrl
## IMAGE_TAG: an easily identifiable tag for your docker image
#export IMAGE_REPO_TAG=xdrl
## IMAGE_URI: the complete URI location for Cloud Container Registry
#export IMAGE_URI=gcr.io/$PROJECT_ID/$IMAGE_REPO_NAME:$IMAGE_REPO_TAG
## Build the docker Image
#docker build -f Dockerfile.docker -t $IMAGE_URI ./
## Test your docker image locally
##docker run $IMAGE_URI #--num-epochs 1
## very manually docker container
##docker run -it  --entrypoint /bin/bash $IMAGE_URI
## Push to the cloud
#docker push $IMAGE_URI
#
#
#export JOB_NAME=xdrl_custom_${DATASET}_${MODEL_NAME}_${TIER}_${CURRENT_DATE}
#echo "Model: ${JOB_NAME}"
#
#gcloud beta ml-engine jobs submit training $JOB_NAME \
#    --stream-logs \
#    --region $REGION \
#    --master-image-uri $IMAGE_URI \
#    --scale-tier $TIER \
#    -- \
#    --job-dir=$MODEL_DIR
