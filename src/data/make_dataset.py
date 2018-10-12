# -*- coding: utf-8 -*-
import click
import logging
import os
from pathlib import Path
from dotenv import find_dotenv, load_dotenv
from .datasets import available_raw_datasets, RawDataset
from ..logging import logger

@click.command()
@click.argument('action')
def main(action, raw_datasets=None):
    """Fetch and/or process the raw data

    Raw files are downloaded into .paths.raw_data_path
    Interim files are generated in .paths.interim_data_path
    Processed data files are saved in .paths.processed_data_path

    action: {'fetch', 'unpack', 'process'}

    """

    if raw_datasets is None:
        raw_datasets = available_raw_datasets()

    for dataset_name in raw_datasets:
        raw_ds = RawDataset.from_name(dataset_name)
        logger.info(f'Running {action} on {dataset_name}')
        if action == 'fetch':
            raw_ds.fetch()
        elif action == 'unpack':
            raw_ds.unpack()
        elif action == 'process':
            ds = raw_ds.process()
            logger.info(f'{dataset_name}: processed data has shape:{ds.data.shape}')


if __name__ == '__main__':
    # not used in this stub but often useful for finding various files
    project_dir = Path(__file__).resolve().parents[2]

    # find .env automagically by walking up directories until it's found, then
    # load up the .env entries as environment variables
    load_dotenv(find_dotenv())

    main()
