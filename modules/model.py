import tensorflow as tf
import tensorflow_transform as tft 
from tensorflow.keras import layers  
import tensorflow_hub as hub
from tfx.components.trainer.fn_args_utils import FnArgs
import tensorflow_text
from typing import Dict, Text


_N_EPOCHS = 30
_BATCH_SIZE = 64

_TEXT_FEATURE_KEYS = [
    'title'
]

_LABEL_KEY = 'sentiment'


# Utility function for renaming the feature
def _transformed_name(key: Text) -> Text:
    """Generate the name of the transformed feature from original name."""
    return key + '_xf'


def _gzip_reader_fn(filenames):
    '''Load compressed dataset
    Args:
      filenames - filenames of TFRecords to load
    Returns:
      TFRecordDataset loaded from the filenames
    '''

    # Load the dataset. Specify the compression type since it is saved as `.gz`
    return tf.data.TFRecordDataset(filenames, compression_type='GZIP')


def _input_fn(file_pattern, 
             tf_transform_output,
             num_epochs=None,
             batch_size=_BATCH_SIZE) -> tf.data.Dataset:
    '''Create batches of features and labels from TF Records
    Args:
      file_pattern - List of files or patterns of file paths containing Example records.
      tf_transform_output - transform output graph
      num_epochs - Integer specifying the number of times to read through the dataset. 
              If None, cycles through the dataset forever.
      batch_size - An int representing the number of records to combine in a single batch.
    Returns:
      A dataset of dict elements, (or a tuple of dict elements and label). 
      Each dict maps feature keys to Tensor or SparseTensor objects.
    '''
    
    # Get post_transform feature spec
    transform_feature_spec = (
        tf_transform_output.transformed_feature_spec().copy())
    
    # create batches of data
    dataset = tf.data.experimental.make_batched_features_dataset(
        file_pattern=file_pattern,
        batch_size=batch_size,
        features=transform_feature_spec,
        reader=_gzip_reader_fn,
        num_epochs=num_epochs,
        label_key=_transformed_name(_LABEL_KEY))
    return dataset


def _get_serve_tf_examples_fn(model, tf_transform_output):
    
    model.tft_layer = tf_transform_output.transform_features_layer()
    
    @tf.function
    def serve_tf_examples_fn(serialized_tf_examples):
        
        feature_spec = tf_transform_output.raw_feature_spec()
        
        feature_spec.pop(_LABEL_KEY)
        
        parsed_features = tf.io.parse_example(serialized_tf_examples, feature_spec)
        
        transformed_features = model.tft_layer(parsed_features)
        
        # get predictions using the transformed features
        return model(transformed_features)
        
    return serve_tf_examples_fn


def _raw_input(model, tf_transform_output):

  model.tft_layer = tf_transform_output.transform_features_layer()

  @tf.function
  def _fn_raw_input(features: Dict[Text, tf.Tensor]):
    feature_spec = tf_transform_output.raw_feature_spec()

    feature_spec.pop(_LABEL_KEY)

    output_features = {}

    for key, spec in feature_spec.items():
      if isinstance(spec, tf.io.VarLenFeature):
        output_features[key] = tf.sparse.from_dense(features[key])
      else:
        output_features[key] = features[key]

    transformed_features = model.tft_layer(output_features)

    outputs = model(transformed_features) 
    return {
      'probabilities': outputs,
      'label_key': tf.argmax(outputs, axis=1),
      'prediction_confidence': tf.reduce_max(outputs, axis=1)
      }

  return _fn_raw_input


def model_builder(hp):
    '''
    Builds the model and sets up the hyperparameters to tune.
    Args:
      hp - Keras tuner object
    Returns:
      model with hyperparameters to tune
    '''
    
    rate = hp.get("dropout")
    kernel_initializer = hp.get("kernel")
    activation_function = hp.get("activation")
    units_1 = hp.get("units_1")
    units_2 = hp.get("units_2")
    learning_rate = hp.get("learning_rate")

    embed = hub.KerasLayer("https://tfhub.dev/google/universal-sentence-encoder-multilingual/3")
    # embed = hub.KerasLayer("https://tfhub.dev/google/universal-sentence-encoder-multilingual-large/3")
    
    # every feature is just a string, so it is shape = (1,)
    inputs = tf.keras.Input(shape=(1,), name=_transformed_name(_TEXT_FEATURE_KEYS[0]), dtype=tf.string)
    # flatten our tensors to 1 dimension
    reshaped_narrative = tf.reshape(inputs, [-1])
    # nn only works with numbers, so we should transform our inputs into numbers through embedding
    # we are using universal-sentence-encoder-multilingual because our strings are in PT-BR
    x = embed(reshaped_narrative)
    # the output of the embed it is a 512 dimensional vector
    x = tf.keras.layers.Reshape((1,512), input_shape=(1,512))(x)
    # here is a feed foward neural network - ffn
    x = layers.Dense(units_1, activation=activation_function, kernel_initializer=kernel_initializer)(x)
    

    attn_output = layers.MultiHeadAttention(num_heads=2, key_dim=units_1)(x, x, x)
    # dropout for regularization
    attn_output = layers.Dropout(rate)(attn_output)
    
    # add & normalization
    out1 = layers.LayerNormalization(epsilon=1e-7)(x + attn_output)
    # ffn
    ffn_output = layers.Dense(units_1, activation=activation_function, kernel_initializer=kernel_initializer)(out1)
    ffn_output = layers.Dense(units_1, kernel_initializer=kernel_initializer)(ffn_output)
    # dropout for regularization
    ffn_output = layers.Dropout(rate)(ffn_output)
    
    # add & normalization 
    x = layers.LayerNormalization(epsilon=1e-7)(out1 + ffn_output)
    # calculating the average for each patch of the feature map
    x = layers.GlobalAveragePooling1D()(x)
    # dropout for regularization
    x = layers.Dropout(rate)(x)
    # ffn
    x = layers.Dense(units_2, activation=activation_function, kernel_initializer=kernel_initializer)(x)
    # dropout for regularization
    x = layers.Dropout(rate)(x)
    # outputs in 3 classes
    outputs = layers.Dense(3, activation='softmax')(x)
    
    
    model = tf.keras.Model(inputs=inputs, outputs = outputs)

    model.compile(
        loss='sparse_categorical_crossentropy',
        optimizer=tf.keras.optimizers.Adam(learning_rate),
        metrics=[tf.keras.metrics.SparseCategoricalAccuracy()]
    )

    model.summary()

    return model 
    

def run_fn(fn_args: FnArgs) -> None:
    """Defines and trains the model.
    Args:
      fn_args: Holds args as name/value pairs. Refer here for the complete attributes: 
      https://www.tensorflow.org/tfx/api_docs/python/tfx/components/trainer/fn_args_utils/FnArgs#attributes
    """    
    
    # Load the transform output
    tf_transform_output = tft.TFTransformOutput(fn_args.transform_graph_path)
    
    # Create batches of data
    train_set = _input_fn(file_pattern=fn_args.train_files, tf_transform_output=tf_transform_output, num_epochs=_N_EPOCHS, batch_size=_BATCH_SIZE)
    val_set = _input_fn(file_pattern=fn_args.eval_files, tf_transform_output=tf_transform_output, num_epochs=_N_EPOCHS, batch_size=_BATCH_SIZE)
    
    # Load best hyperparameters
    hp = fn_args.hyperparameters.get('values')

    # Build the model
    model = model_builder(hp)
    
    # Train the model
    model.fit(
        x = train_set,
        validation_data = val_set,
        #  steps_per_epoch = 1000, 
        #  validation_steps= 1000,
    )
    
    signatures = {
        'serving_default':
        _get_serve_tf_examples_fn(model, 
                                 tf_transform_output).get_concrete_function(
                                    tf.TensorSpec(
                                    shape=[None],
                                    dtype=tf.string,
                                    name='examples')),
         'predict_raw':
         _raw_input(model, tf_transform_output).get_concrete_function(
                        dict(
                            title=tf.TensorSpec(
                                shape=(None), dtype=tf.string, name='title-prod')
                                ),
                          ), 
    }

    model.save(fn_args.serving_model_dir, save_format='tf', signatures=signatures)