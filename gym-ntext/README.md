

## gcloud configuration

to be able to run in google cloud, we need to configure authentication

create user
````bash 
gcloud iam service-accounts create [ACCOUNT_NAME]
````

assign role

````bash
gcloud projects add-iam-policy-binding [PROJECT_ID] --member "serviceAccount:[ACCOUNT_NAME]@[PROJECT_ID].iam.gserviceaccount.com" --role "roles/owner"
````

get key
````bash
gcloud iam service-accounts keys create [FILENAME].json --iam-account [ACCOUNT_NAME]@[PROJECT_ID].iam.gserviceaccount.com
export GOOGLE_APPLICATION_CREDENTIALS="[FILENAME].json"
````

