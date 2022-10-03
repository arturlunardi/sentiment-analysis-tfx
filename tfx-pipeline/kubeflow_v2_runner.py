import os
from absl import logging
from pipeline import configs
from pipeline import run_pipeline
from tfx.orchestration.kubeflow.v2 import kubeflow_v2_dag_runner


# TFX pipeline produces many output files and metadata. All output data will be
# stored under this OUTPUT_DIR..
OUTPUT_DIR = os.path.join('gs://', configs.GCS_BUCKET_NAME)

# TFX produces two types of outputs, files and metadata.
# - Files will be created under PIPELINE_ROOT directory.
PIPELINE_ROOT = os.path.join(OUTPUT_DIR, 'tfx_pipeline_output',
                             configs.PIPELINE_NAME)

# The last component of the pipeline, "Pusher" will produce serving model under
# SERVING_MODEL_DIR.;
SERVING_MODEL_DIR = os.path.join(PIPELINE_ROOT, 'serving_model')


def run():
    """Define a kubeflow pipeline."""

    # Metadata config. The defaults works work with the installation of
    # KF Pipelines using Kubeflow. If installing KF Pipelines using the
    # lightweight deployment option, you may need to override the defaults.
    # If you use Kubeflow, metadata will be written to MySQL database inside
    # Kubeflow cluster.
    
    runner_config = kubeflow_v2_dag_runner.KubeflowV2DagRunnerConfig(
      default_image=configs.PIPELINE_IMAGE)

    dsl_pipeline = run_pipeline.create_pipeline(pipeline_name=configs.PIPELINE_NAME,
                        pipeline_root=PIPELINE_ROOT,
                        data_path=configs.DATA_PATH,
                        preprocessing_module=configs.TRANSFORM_MODULE_FILE,
                        tuner_path=configs.TUNER_MODULE_PATH,
                        training_module=configs.TRAIN_MODULE_FILE,
                        serving_model_dir=SERVING_MODEL_DIR,
                        # TODO(step 9): (Optional) Uncomment below to use Cloud AI Platform.
                        ai_platform_training_args=configs.GCP_AI_PLATFORM_TRAINING_ARGS,
                        # TODO(step 9): (Optional) Uncomment below to use Cloud AI Platform.
                        ai_platform_serving_args=configs.GCP_AI_PLATFORM_SERVING_ARGS,
                        )
    
    runner = kubeflow_v2_dag_runner.KubeflowV2DagRunner(config=runner_config)

    runner.run(pipeline=dsl_pipeline)


if __name__ == '__main__':
    logging.set_verbosity(logging.INFO)
    run()