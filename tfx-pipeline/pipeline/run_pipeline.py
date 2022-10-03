from tfx.components import CsvExampleGen
from tfx.components import SchemaGen
from tfx.components import ExampleValidator
from tfx.components import StatisticsGen
# from tfx.components import ImporterNode
from tfx.v1.dsl import Importer
from tfx.components import Transform
from tfx.components import Trainer
from tfx.components import Evaluator
from tfx.dsl.components.common.resolver import Resolver
from tfx.dsl.input_resolution.strategies.latest_blessed_model_strategy import LatestBlessedModelStrategy
from tfx.components import Pusher
import os
from tfx.orchestration import pipeline
import tensorflow_model_analysis as tfma
from tfx import v1 as tfx
from typing import List, Optional, Text, Dict
from tfx.proto import example_gen_pb2
from tfx.proto import trainer_pb2
from tfx.types import standard_artifacts
from tfx.types import Channel
from tfx.types.standard_artifacts import Model, ModelBlessing
from ml_metadata.proto import metadata_store_pb2


def create_pipeline(
    pipeline_name: Text,
    pipeline_root: Text,
    data_path: Text,
    preprocessing_module: Text,
    tuner_path: Text,
    training_module: Text,
    serving_model_dir: Text,
    beam_pipeline_args: Optional[List[Text]] = None,
    ai_platform_training_args: Optional[Dict[Text, Text]] = None,
    ai_platform_serving_args: Optional[Dict[Text, Text]] = None,
    enable_cache: Optional[bool] = False,
    metadata_connection_config: Optional[metadata_store_pb2.ConnectionConfig] = None,
    ) -> pipeline.Pipeline:
    """Implements the pipeline with TFX."""

    # initialize components list
    components = []

    # Brings data into the pipeline or otherwise joins/converts training data.
    output = example_gen_pb2.Output(
        split_config=example_gen_pb2.SplitConfig(splits=[
            example_gen_pb2.SplitConfig.Split(
                name='train', hash_buckets=3),
            example_gen_pb2.SplitConfig.Split(
                name='eval', hash_buckets=1)
        ]))

    example_gen = CsvExampleGen(
        input_base=data_path, output_config=output)
    components.append(example_gen)

    # Computes statistics over data for visualization and example validation.
    statistics_gen = StatisticsGen(
        examples=example_gen.outputs['examples'])
    components.append(statistics_gen)

    # Generates schema based on statistics files.
    schema_gen = SchemaGen(
        statistics=statistics_gen.outputs['statistics'], infer_feature_shape=True)
    components.append(schema_gen)

    # Performs anomaly detection based on statistics and data schema.
    example_validator = ExampleValidator(  # pylint: disable=unused-variable
        statistics=statistics_gen.outputs['statistics'],
        schema=schema_gen.outputs['schema'])
    components.append(example_validator)

    # Performs transformations and feature engineering in training and serving.
    transform = Transform(
        examples=example_gen.outputs['examples'],
        schema=schema_gen.outputs['schema'],
        module_file=preprocessing_module,
    )

    components.append(transform)

    hparams_importer = Importer(
        source_uri=tuner_path,
        artifact_type=standard_artifacts.HyperParameters).with_id('import_hparams')

    components.append(hparams_importer)

    trainer_args = dict(
        module_file=training_module,
        examples=transform.outputs['transformed_examples'],
        hyperparameters=hparams_importer.outputs['result'],
        transform_graph=transform.outputs['transform_graph'],
        schema=schema_gen.outputs['schema'],
        train_args=trainer_pb2.TrainArgs(splits=['train']),
        eval_args=trainer_pb2.EvalArgs(splits=['eval']),
    )

    if ai_platform_training_args is not None:
        trainer_args['custom_config'] = {
            tfx.extensions.google_cloud_ai_platform.TRAINING_ARGS_KEY:
                ai_platform_training_args,
        }   
        
        trainer = tfx.extensions.google_cloud_ai_platform.Trainer(**trainer_args)
    else:
        trainer = Trainer(**trainer_args)

    components.append(trainer)

    model_resolver = Resolver(
        strategy_class=LatestBlessedModelStrategy,
        model=Channel(type=Model),
        model_blessing=Channel(
            type=ModelBlessing))

    components.append(model_resolver)

    # Uses TFMA to compute a evaluation statistics over features of a model and
    # perform quality validation of a candidate model (compared to a baseline).
    eval_config = tfma.EvalConfig(
    model_specs=[tfma.ModelSpec(label_key='sentiment')],
    slicing_specs=[tfma.SlicingSpec()],
    metrics_specs=[
        tfma.MetricsSpec(metrics=[
            
            tfma.MetricConfig(class_name='ExampleCount'),
            tfma.MetricConfig(class_name='AUC'),
            tfma.MetricConfig(class_name='FalsePositives'),
            tfma.MetricConfig(class_name='TruePositives'),
            tfma.MetricConfig(class_name='FalseNegatives'),
            tfma.MetricConfig(class_name='TrueNegatives'),
            tfma.MetricConfig(
                  class_name='SparseCategoricalAccuracy',
                  threshold=tfma.MetricThreshold(
                      value_threshold=tfma.GenericValueThreshold(
                          lower_bound={'value': 0}),
                      # Change threshold will be ignored if there is no
                      # baseline model resolved from MLMD (first run).
                      change_threshold=tfma.GenericChangeThreshold(
                          direction=tfma.MetricDirection.HIGHER_IS_BETTER,
                          absolute={'value': -1e-10}
                      )
                  )
            )
        ])
    ])

    evaluator = Evaluator(
        examples=example_gen.outputs['examples'],
        model=trainer.outputs['model'],
        baseline_model=model_resolver.outputs['model'],
        eval_config=eval_config)

    components.append(evaluator)

    pusher_args = {
        'model': trainer.outputs['model'],
        'model_blessing': evaluator.outputs['blessing'],
    }

    if ai_platform_serving_args is not None:
        pusher_args['custom_config'] = {
            tfx.extensions.google_cloud_ai_platform.experimental
            .PUSHER_SERVING_ARGS_KEY:
                ai_platform_serving_args
        }
        
        pusher = tfx.extensions.google_cloud_ai_platform.Pusher(**pusher_args)
    else:
        pusher_args['push_destination'] = tfx.proto.PushDestination(
        filesystem=tfx.proto.PushDestination.Filesystem(
            base_directory=serving_model_dir))
        # Pushes the model to a file destination if check passed.
        pusher = Pusher(**pusher_args)

    components.append(pusher)
 
    return tfx.dsl.Pipeline(
        pipeline_name=pipeline_name,
        pipeline_root=pipeline_root,
        components=components,
        beam_pipeline_args=beam_pipeline_args,
        enable_cache=enable_cache,
        metadata_connection_config=metadata_connection_config,
    )