import os
import tempfile
import json
import apache_beam as beam
import tensorflow as tf
import tensorflow_transform as tft
import tensorflow_transform.beam as tft_beam
from tensorflow_transform.tf_metadata import dataset_metadata
from tensorflow_transform.tf_metadata import dataset_schema
from apache_beam.options.pipeline_options import PipelineOptions
from apache_beam.options.pipeline_options import SetupOptions


def gather_raw_metadata(numerical_feats, categorical_feats):

    raw_data_feature_spec = dict([(name, tf.io.FixedLenFeature([], tf.float32))
                                  for name in numerical_feats] +
                                 [(name, tf.io.FixedLenFeature([], tf.string))
                                  for name in categorical_feats] +
                                 [("label", tf.io.FixedLenFeature([], tf.float32))])

    raw_data_meta_data = dataset_metadata.DatasetMetadata(
        dataset_schema.from_feature_spec(raw_data_feature_spec))
    return raw_data_meta_data


def transform_data(train_data_file, test_data_file, working_dir):
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

    numerical_feats = [
        "startCountTotal",
        "purchaseCountTotal",
        "globalStartCountTotal",
        "globalPurchaseCountTotal"
    ]

    categorical_feats = [
        "country",
        "sourceGameId",
        "platform"
    ]
    

    def preprocessing_fn(inputs):
        """Preprocess input columns into transformed columns."""
        outputs = {}

        for key in numerical_feats:
            outputs[key] = tf.cast(tft.bucketize(inputs[key], 20), tf.float32) / 20.0 - 0.5

        outputs["campaignCost_mod"] = inputs["campaignCost"]/100.0

        inputs["game_zone"] = tf.string_join([inputs["sourceGameId"], inputs["zone"]], separator="_")
        inputs["game_campaignId"] = tf.string_join([inputs["sourceGameId"], inputs["campaignId"]], separator="_")

        for key in categorical_feats + ["game_zone", "game_campaignId"]:
            vocab = tft.vocabulary(inputs[key], vocab_filename=key, frequency_threshold=100)
            outputs[key] = tft.apply_vocabulary(inputs[key], vocab, default_value=0)

        outputs["label"] = inputs["label"]
        outputs["key"] = inputs["key"]

        return outputs

    # Input schema definition
    RAW_DATA_METADATA = gather_raw_metadata(numerical_feats + ["campaignCost"],
                                            categorical_feats + ["zone", "campaignId", "key"])

 
    # pipeline args to read from gcs, currently unused because reading local file
    pipeline_args=[
      '--runner=DirectRunner',
      '--project=unity-ads-ds-prd',
 #     '--staging_location=gs://unity-ads-ds-prd-users/villew/promo/staging',
 #     '--temp_location=gs://unity-ads-ds-prd-users/villew/promo/temp',
      '--job_name=transform-promo-data-to-tf-records'
    ]
    pipeline_options = PipelineOptions(pipeline_args)
    pipeline_options.view_as(SetupOptions).save_main_session = True

    # create a beam pipeline
    with beam.Pipeline(options=pipeline_options) as pipeline:
        with tft_beam.Context(temp_dir=tempfile.mkdtemp()):
            raw_data = (
                    pipeline
                    | 'ReadTrainData' >> beam.io.ReadFromParquet(train_data_file))

            # Combine data and schema into a dataset tuple.
            raw_dataset = (raw_data, RAW_DATA_METADATA)
            transformed_dataset, transform_fn = (
                    raw_dataset | tft_beam.AnalyzeAndTransformDataset(preprocessing_fn))
            transformed_data, transformed_metadata = transformed_dataset
            transformed_data_coder = tft.coders.ExampleProtoCoder(
                transformed_metadata.schema)

            # write to tf record
            _ = (
                    transformed_data
                    | 'EncodeTrainData' >> beam.Map(transformed_data_coder.encode)
                    | 'WriteTrainData' >> beam.io.WriteToTFRecord(
                os.path.join(working_dir, "train_tfrecord")))


            # Now apply transform function to test data.
            raw_test_data = (
                    pipeline
                    | 'ReadTestData' >> beam.io.ReadFromParquet(test_data_file))

            raw_test_dataset = (raw_test_data, RAW_DATA_METADATA)

            transformed_test_dataset = (
                    (raw_test_dataset, transform_fn) | tft_beam.TransformDataset())

            # Don't need transformed data schema, it's the same as before.
            transformed_test_data, _ = transformed_test_dataset

            _ = (
                    transformed_test_data
                    | 'EncodeTestData' >> beam.Map(transformed_data_coder.encode)
                    | 'WriteTestData' >> beam.io.WriteToTFRecord(
                os.path.join(working_dir, "test_tfrecord")))

            # Will write a SavedModel and metadata to working_dir, which can then
            # be read by the tft.TFTransformOutput class.
            _ = (
                    transform_fn
                    | 'WriteTransformFn' >> tft_beam.WriteTransformFn(working_dir))


def main():
    #output_dir = os.path.join("gs://unity-ads-ds-prd-users/villew/promo", "output4")
    output_dir = os.path.join("gs://unity-ads-ds-prd-users/nimas/promo", "tft/output4-down-sampled")
    train_data_file = 'gs://unity-ads-ds-prd-users/nimas/promo/output4-down-sampled/*.parquet'
    #train_data_file = 'gs://unity-ads-iap-valuation-data-test/iap-promo-1.0.0-1e4/date=2019-09-12/hour=22/parquet/split=training/'
    test_data_file = 'gs://unity-ads-iap-valuation-data-test/iap-promo-1.0.0-1e4/date=2019-09-12/hour=22/parquet/split=test/'
    transform_data(train_data_file, test_data_file, output_dir)


if __name__ == '__main__':
    main()
