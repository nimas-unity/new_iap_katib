import os
import tempfile
import apache_beam as beam
import tensorflow as tf
import tensorflow_transform as tft
import tensorflow_transform.beam as tft_beam
from tensorflow_transform.tf_metadata import dataset_metadata
from tensorflow_transform.tf_metadata import dataset_schema
from apache_beam.options.pipeline_options import PipelineOptions
from apache_beam.options.pipeline_options import SetupOptions


def _get_raw_metadata(input_features):
    raw_data_feature_spec = {}
    for name in input_features["numerical_default_encoding"] + input_features["numerical_special"]:
        raw_data_feature_spec[name] = tf.io.FixedLenFeature([], tf.float32)

    for name in input_features["categorical_default_encoding"] + input_features["categorical_special"]:
        raw_data_feature_spec[name] = tf.io.FixedLenFeature([], tf.string)

    if "label" in input_features:
        raw_data_feature_spec[input_features["label"]] = tf.io.FixedLenFeature([], tf.float32)

    raw_data_meta_data = dataset_metadata.DatasetMetadata(dataset_schema.from_feature_spec(raw_data_feature_spec))
    return raw_data_meta_data


def transform_data(input_features, preprocessing_fn, pipeline_args,
                   train_data_file, cv_data_file, test_data_file, working_dir):
    """Transform the data and write out as a TFRecord of Example protos.
    Read in the data using the parquet io, and transform it using a
    preprocessing pipeline that scales numeric data and converts categorical data
    from strings to int64 values indices, by creating a vocabulary for each
    category.
    Args:
      train_data_file: File containing training data
      test_data_file: File containing test data
      feature_config: named tuple with feature types
      working_dir: Directory to write transformed data and metadata to
    """

    # Input schema definition
    RAW_DATA_METADATA = _get_raw_metadata(input_features)

    # pipeline args to read from gcs, currently unused because reading local file
    pipeline_options = PipelineOptions(pipeline_args)
    pipeline_options.view_as(SetupOptions).save_main_session = True

    # create a beam pipeline
    with beam.Pipeline(options=pipeline_options) as pipeline:
        # Needs to be GCS location if the process is running on Dataflow, otherwise it can't share model files
        temp_dir = pipeline_options.get_all_options().get('temp_location') or tempfile.mkdtemp()
        with tft_beam.Context(temp_dir=temp_dir):
            raw_data = (
                    pipeline
                    | 'ReadTrainData' >> beam.io.ReadFromParquet(train_data_file))

            # Combine data and schema into a dataset tuple.
            raw_dataset = (raw_data, RAW_DATA_METADATA)
            transformed_dataset, transform_fn = (
                    raw_dataset | tft_beam.AnalyzeAndTransformDataset(preprocessing_fn))
            transformed_data, transformed_metadata = transformed_dataset
            transformed_data_coder = tft.coders.ExampleProtoCoder(transformed_metadata.schema)

            # write to tf record
            _ = (
                    transformed_data
                    | 'EncodeTrainData' >> beam.Map(transformed_data_coder.encode)
                    | 'WriteTrainData' >> beam.io.WriteToTFRecord(os.path.join(working_dir, "train_tfrecord")))

            def encode_data(data_path, prefix, output_filename):
                # Apply transform function to test data.
                raw_data = (
                        pipeline
                        | 'ReadData'+prefix >> beam.io.ReadFromParquet(data_path))

                raw_dataset = (raw_data, RAW_DATA_METADATA)

                transformed_dataset = (
                        (raw_dataset, transform_fn) | 'Transform'+prefix >> tft_beam.TransformDataset())

                # Don't need transformed data schema, it's the same as before.
                transformed_data, _ = transformed_dataset

                _ = (
                        transformed_data
                        | 'EncodeData'+prefix >> beam.Map(transformed_data_coder.encode)
                        | 'WriteData'+prefix >> beam.io.WriteToTFRecord(os.path.join(working_dir, output_filename)))

            encode_data(cv_data_file, "-cv", "cv_tfrecord")
            encode_data(test_data_file, "-test", "test_tfrecord")

            # Will write a SavedModel and metadata to working_dir, which can then
            # be read by the tft.TFTransformOutput class.
            _ = (
                    transform_fn
                    | 'WriteTransformFn' >> tft_beam.WriteTransformFn(working_dir))

