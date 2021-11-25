#!/usr/bin/env python
"""
Download from W&B the raw dataset and apply some basic data cleaning, exporting the result to a new artifact
"""
import argparse
import logging
import os
import tempfile

import pandas as pd

import wandb

logging.basicConfig(level=logging.INFO, format="%(asctime)-15s %(message)s")
logger = logging.getLogger()


def clean_df(df: pd.DataFrame, min_price: float, max_price: float) -> pd.DataFrame:
    """Clean raw DataFrame by removing outliers and converting `last_review` column type to datetime.

    Outliers are defined by the rows whose `price` feature is not comprised within given price range.

    Args:
        df (pd.DataFrame): Raw Pandas DataFrame.
        min_price (float): Price lower-bound that defines inliers. 
        max_price (float): Price upper-bound that defines inliers.

    Returns:
        pd.DataFrame: Cleaned Pandas DataFrame
    """
    # Remove outliers based on price
    idx = df['price'].between(min_price, max_price)
    df = df[idx].copy()

    # Convert last_review to datetime
    df['last_review'] = pd.to_datetime(df['last_review'])

    # Remove outliers based on longitude and latitude
    idx = df['longitude'].between(-74.25, -
                                  73.50) & df['latitude'].between(40.5, 41.2)
    df = df[idx].copy()
    return df


def go(args):
    run = wandb.init(job_type="basic_cleaning")
    run.config.update(args)

    # Download input artifact from W&B
    logger.info("Downloading raw DataFrame from W&B")
    artifact_local_path = run.use_artifact(args.input_artifact).file()

    # Read and clean raw dataframe
    logger.info("Cleaning DataFrame")
    df = pd.read_csv(artifact_local_path)
    df_clean = clean_df(df, args.min_price, args.max_price)

    with tempfile.TemporaryDirectory() as tmp_dir:
        logger.info("Saving cleaned DataFrame on W&B")
        df_clean.to_csv(os.path.join(tmp_dir, "clean_sample.csv"), index=False)

        # Save cleaned data to W&B
        artifact = wandb.Artifact(
            args.output_artifact,
            type=args.output_type,
            description=args.output_description,
        )
        artifact.add_file(os.path.join(tmp_dir, "clean_sample.csv"))
        run.log_artifact(artifact)


if __name__ == "__main__":

    parser = argparse.ArgumentParser(description="A very basic data cleaning")

    parser.add_argument(
        "--input_artifact",
        type=str,
        help="Input artifact name on W&B.",
        required=True
    )

    parser.add_argument(
        "--output_artifact",
        type=str,
        help="Name for the W&B artifact that will be created.",
        required=True
    )

    parser.add_argument(
        "--output_type",
        type=str,
        help="Type of the artifact to create",
        required=True
    )

    parser.add_argument(
        "--output_description",
        type=str,
        help="Description for the artifact",
        required=True
    )

    parser.add_argument(
        "--min_price",
        type=float,
        help="Price lower bound for inliers.",
        required=True
    )

    parser.add_argument(
        "--max_price",
        type=float,
        help="Price lower bound for inliers.",
        required=True
    )

    args = parser.parse_args()

    go(args)
