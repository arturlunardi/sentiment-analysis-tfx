import os
from absl import logging
from pipeline import configs
from pipeline import pipeline
from tfx import v1 as tfx

# TFX pipeline produces many output files and metadata. All output data will be
# stored under this OUTPUT_DIR.
# NOTE: It is recommended to have a separated OUTPUT_DIR which is *outside* of
#       the source code structure. Please change OUTPUT_DIR to other location
#       where we can store outputs of the pipeline.
OUTPUT_DIR = '.'

# TFX produces two types of outputs, files and metadata.
# - Files will be created under PIPELINE_ROOT directory.
# - Metadata will be written to SQLite database in METADATA_PATH.
PIPELINE_ROOT = os.path.join(OUTPUT_DIR, 'tfx_pipeline_output',
                             configs.PIPELINE_NAME)

METADATA_PATH = os.path.join(OUTPUT_DIR, 'tfx_metadata', configs.PIPELINE_NAME,
                             'metadata.db')

# The last component of the pipeline, "Pusher" will produce serving model under
# SERVING_MODEL_DIR.
SERVING_MODEL_DIR = os.path.join(PIPELINE_ROOT, 'serving_model')


# Specifies data file directory. DATA_PATH should be a directory containing CSV
# files for CsvExampleGen in this example. By default, data files are in the
# `data` directory.
# NOTE: If you upload data files to GCS(which is recommended if you use
#       Kubeflow), you can use a path starting "gs://YOUR_BUCKET_NAME/path" for
#       DATA_PATH. For example,
#       DATA_PATH = 'gs://bucket/chicago_taxi_trips/csv/'.

def run():
  """Define a local pipeline."""

  tfx.orchestration.LocalDagRunner().run(
      pipeline.create_pipeline(
          pipeline_name=configs.PIPELINE_NAME,
          pipeline_root=PIPELINE_ROOT,
          data_path=configs.LOCAL_DATA_PATH,
          preprocessing_module=configs.LOCAL_TRANSFORM_MODULE_FILE,
          tuner_path=configs.LOCAL_TUNER_MODULE_PATH,
          training_module=configs.LOCAL_TRAIN_MODULE_FILE,
          serving_model_dir=SERVING_MODEL_DIR,
          # TODO(step 7): (Optional) Uncomment here to use provide GCP related
          #               config for BigQuery with Beam DirectRunner.
          # beam_pipeline_args=configs.
          # BIG_QUERY_WITH_DIRECT_RUNNER_BEAM_PIPELINE_ARGS,
          metadata_connection_config=tfx.orchestration.metadata
          .sqlite_metadata_connection_config(METADATA_PATH)
          )
        )


if __name__ == '__main__':
  logging.set_verbosity(logging.INFO)
  run()