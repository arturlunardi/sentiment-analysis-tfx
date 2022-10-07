import os

# Pipeline name will be used to identify this pipeline.
PIPELINE_NAME = 'sentiment-analysis-tfx'

# GCP related configs.

# Following code will retrieve your GCP project. You can choose which project 
# to use by setting GOOGLE_CLOUD_PROJECT environment variable.
try:
    import google.auth  # pylint: disable=g-import-not-at-top  # pytype: disable=import-error
    try:
        _, GOOGLE_CLOUD_PROJECT = google.auth.default()
    except google.auth.exceptions.DefaultCredentialsError:
        GOOGLE_CLOUD_PROJECT = ''
except ImportError:
    GOOGLE_CLOUD_PROJECT = ''

# Specify your GCS bucket name here. You have to use GCS to store output files 
# when running a pipeline with Kubeflow Pipeline on GCP or when running a job
# using Dataflow. Default is '<gcp_project_name>-kubeflowpipelines-default'.
# This bucket is created automatically when you deploy KFP from marketplace.
# GCS_BUCKET_NAME = GOOGLE_CLOUD_PROJECT + '-kubeflowpipelines-default'
GCS_BUCKET_NAME = GOOGLE_CLOUD_PROJECT + '-vertex-default'

GOOGLE_CLOUD_REGION = 'us-central1'

# GCP
TRANSFORM_MODULE_FILE = 'gs://{}/{}/modules/preprocessing.py'.format(GCS_BUCKET_NAME, PIPELINE_NAME)
TRAIN_MODULE_FILE = 'gs://{}/{}/modules/model.py'.format(GCS_BUCKET_NAME, PIPELINE_NAME)
TUNER_MODULE_PATH = 'gs://{}/{}/best_hyperparameters/'.format(GCS_BUCKET_NAME, PIPELINE_NAME)
DATA_PATH = 'gs://{}/{}/data/'.format(GCS_BUCKET_NAME, PIPELINE_NAME)
LABEL_ENCODER_FILE = 'gs://{}/{}/modules/label_encoder.pkl'.format(GCS_BUCKET_NAME, PIPELINE_NAME)

# LOCAL
LOCAL_TRANSFORM_MODULE_FILE = os.path.abspath(os.path.join(os.path.dirname(os.path.abspath(__file__)), r'../..', 'modules', 'preprocessing.py'))
LOCAL_TRAIN_MODULE_FILE = os.path.abspath(os.path.join(os.path.dirname(os.path.abspath(__file__)), r'../..', 'modules', 'model.py'))
LOCAL_TUNER_MODULE_PATH = os.path.abspath(os.path.join(os.path.dirname(os.path.abspath(__file__)), r'../..', 'modules', 'best_hyperparameters'))
LOCAL_DATA_PATH = os.path.abspath(os.path.join(os.path.dirname(os.path.abspath(__file__)), r'../..', 'modules', 'data'))
LOCAL_LABEL_ENCODER_FILE = os.path.abspath(os.path.join(os.path.dirname(os.path.abspath(__file__)), r'../..', 'modules', 'label_encoder.pkl'))

# Following image will be used to run pipeline components run if Kubeflow
# Pipelines used.
# This image will be automatically built by CLI if we use --build-image flag.
PIPELINE_IMAGE = f'gcr.io/{GOOGLE_CLOUD_PROJECT}/{PIPELINE_NAME}'


# (Optional) Uncomment below to use AI Platform training.
GCP_AI_PLATFORM_TRAINING_ARGS = {
    'project': GOOGLE_CLOUD_PROJECT,
    'region': GOOGLE_CLOUD_REGION,
    # Starting from TFX 0.14, training on AI Platform uses custom containers:
    # https://cloud.google.com/ml-engine/docs/containers-overview
    # You can specify a custom container here. If not specified, TFX will use
    # a public container image matching the installed version of TFX.
    # (Optional) Set your container name below.
    'masterConfig': {
      'imageUri': PIPELINE_IMAGE
    },
    # Note that if you do specify a custom container, ensure the entrypoint
    # calls into TFX's run_executor script (tfx/scripts/run_executor.py)
}

# A dict which contains the serving job parameters to be passed to Google
# Cloud AI Platform. For the full set of parameters supported by Google Cloud AI
# Platform, refer to
# https://cloud.google.com/ml-engine/reference/rest/v1/projects.models
# (Optional) Uncomment below to use AI Platform serving.
GCP_AI_PLATFORM_SERVING_ARGS = {
    'model_name': PIPELINE_NAME.replace('-','_'),  # '-' is not allowed.
    'project_id': GOOGLE_CLOUD_PROJECT,
    # The region to use when serving the model. See available regions here:
    # https://cloud.google.com/ml-engine/docs/regions
    # Note that serving currently only supports a single region:
    # https://cloud.google.com/ml-engine/reference/rest/v1/projects.models#Model  # pylint: disable=line-too-long
    'regions': [GOOGLE_CLOUD_REGION],
}