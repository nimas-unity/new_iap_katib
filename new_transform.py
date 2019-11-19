import os
import tensorflow as tf
import tensorflow_transform as tft

from transform_utils import transform_data


def _default_preprocessing_fn(inputs, input_features):
    outputs = {}

    for key in input_features["numerical_default_encoding"]:
        outputs[key] = tf.cast(tft.bucketize(inputs[key], 20), tf.float32) / 20.0 - 0.5

    for key in input_features["categorical_default_encoding"]:
        vocab = tft.vocabulary(inputs[key], vocab_filename=key, frequency_threshold=100)
        outputs[key] = tft.apply_vocabulary(inputs[key], vocab, default_value=0)

    if "label" in input_features:
        outputs["label"] = inputs[input_features["label"]]

    return outputs


INPUT_FEATURES = {
    "numerical_default_encoding": [  # Will be encoded with the default pre-processing function
        "startCountTotal",
        "purchaseCountTotal",
        "globalStartCountTotal",
        "globalPurchaseCountTotal"
    ],
    "numerical_special": [
        "campaignCost"
    ],
    "categorical_default_encoding": [  # Will be encoded with the default pre-processing function
        "country",
        "sourceGameId",
        "platform"
    ],
    "categorical_special": [
        "zone",
        "campaignId",
        "key"
    ],
    "label": "label"
}


def preprocessing_fn(inputs, input_features):
    """Preprocess input columns into transformed columns."""

    outputs = _default_preprocessing_fn(inputs, input_features)

    outputs["campaignCost_mod"] = inputs["campaignCost"] / 100.0

    inputs["game_zone"] = tf.string_join([inputs["sourceGameId"], inputs["zone"]], separator="_")
    inputs["game_campaignId"] = tf.string_join([inputs["sourceGameId"], inputs["campaignId"]], separator="_")

    for key in ["game_zone", "game_campaignId"]:
        vocab = tft.vocabulary(inputs[key], vocab_filename=key, frequency_threshold=100)
        outputs[key] = tft.apply_vocabulary(inputs[key], vocab, default_value=0)

    outputs["key"] = inputs["key"]

    return outputs


def main():

    pipeline_args=[
      '--runner=DataFlowRunner',
      '--project=unity-ads-ds-prd',
      '--staging_location=gs://unity-ads-ds-prd-users/villew/promo/staging',
      '--temp_location=gs://unity-ads-ds-prd-users/villew/promo/temp',  # Must be GCS location for DataFlowRunner
      '--job_name=transform-promo-data-to-tf-records-1',
      '--setup_file=./setup.py',  # Needed for DataFlowRunner
      '--worker_machine_type=n1-standard-16'
    ]
    print('+++++++++++++++++++++++++++++++nima-1+++++++++++++++++++++++++++++++')

    output_dir = os.path.join("gs://unity-ads-ds-prd-users/nimas/promo/tft", "output7")

    promo_base = "gs://unity-ads-ds-prd-users/nimas/new_ville_data/promo_data/"
    train_data_file = os.path.join(promo_base, "split=train/")
    cv_data_file = os.path.join(promo_base, "split=eval/")
    test_data_file = os.path.join(promo_base, "split=test/")

    transform_data(INPUT_FEATURES, lambda inputs: preprocessing_fn(inputs, INPUT_FEATURES), pipeline_args,
                   train_data_file, cv_data_file, test_data_file, output_dir)


if __name__ == '__main__':
    print('+++++++++++++++++++++++++++++++nima-0+++++++++++++++++++++++++++++++')
    main()
