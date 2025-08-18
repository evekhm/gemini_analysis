# Latency Analysis

## Prerequisites

### Enable Gemini Requests Logging
Vertex AI can log samples of requests and responses for Gemini and supported partner models.
The logs are saved to a BigQuery table for viewing and analysis.

To enable logging, follow these [instructions](https://cloud.google.com/vertex-ai/generative-ai/docs/multimodal/request-response-logging)


Needs to be done to each model being used:
```shell
MODEL="gemini-2.0-flash-lite"
# MODEL="gemini-2.0-flash"
curl -X POST \
-H "Authorization: Bearer $(gcloud auth print-access-token)" \
-H "Content-Type: application/json; charset=utf-8" \
-d @request.json \
"https://$REGION-aiplatform.googleapis.com/v1beta1/projects/$PROJECT_ID/locations/$REGION/publishers/google/models/$MODEL:setPublisherModelConfig"
```

> request.json
```json
{
  "publisherModelConfig": {
     "loggingConfig": {
       "enabled": true,
       "samplingRate": 1,
       "bigqueryDestination": {
         "outputUri": "bq://PROJECT_ID.DATASET.TABLENAME"
       },
       "enableOtelLogging": true
     }
   }
 }
```

### Install libraries

```shell
python -m venv venv
source venv/bin/activate
```

Install libraries:

```shell
pip install -r requirements.txt
```

### Set environment variables
```shell
export PROJECT_ID="..."
export DATASET="..."
export GEMINI_LOG_TABLE="..."  # configured for logging
```

## Generate Analysis of Gemini Level Logs

Usage examples:
```shell
# Last 10 days 
python gemini_logs.py -d 10

# Last 10 days ending at a specific time
python gemini_logs.py -d 10 -e "2025-01-15 23:59:59"

# Also works with explicit start/end (ignores -d if not specified)
python gemini_logs.py -s "2025-01-01 00:00:00" -e "2025-01-31 23:59:59"

# All data starting from a specific date till current timestamp
python gemini_logs.py --start "2025-01-15 12:00:00"
```


```shell
python gemini_logs.py --start "2025-08-14 00:00:00" --end "2025-08-20 23:59:59"
```

## Application Logs

Usage examples:
```shell
# Last 10 days 
python prediction_logs.py -d 10

# Last 10 days ending at a specific time
python prediction_logs.py -d 10 -e "2025-01-15 23:59:59"

# Also works with explicit start/end (ignores -d if not specified)
python prediction_logs.py -s "2025-01-01 00:00:00" -e "2025-01-31 23:59:59"

# All data starting from a specific date till current timestamp
python prediction_logs.py --start "2025-01-15 12:00:00"
```



