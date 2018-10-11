# -*- coding: utf-8 -*-
import click
import logging
import os
from pathlib import Path
from dotenv import find_dotenv, load_dotenv
from .datasets import available_datasets, load_dataset
from .dset import RawDataset
from ..paths import data_path
from ..logging import logger

@click.command()
@click.argument('action')
def main(action, datasets=None):
    """Fetch and/or process the raw data

    Runs data processing scripts to turn raw data from (../raw) into

    Raw files are downloaded into `project_dir`/data/raw
    Interim files are generated in `project_dir`/data/interim

    action: {'fetch', 'process'}

    """
    logger.info(f'Dataset: running {action}')

    if datasets is None:
        datasets, _ = available_datasets(keys_only=False)

    for dataset_name in datasets:
        raw_ds = RawDataset.from_dict(datasets[dataset_name])
        if action == 'fetch':
            raw_ds.fetch()
        elif action == 'unpack':
            raw_ds.fetch()
            raw_ds.unpack()
        elif action == 'process':
            raw_ds.fetch()
            raw_ds.unpack()
            ds = raw_ds.process()


if __name__ == '__main__':
    # not used in this stub but often useful for finding various files
    project_dir = Path(__file__).resolve().parents[2]

    # find .env automagically by walking up directories until it's found, then
    # load up the .env entries as environment variables
    load_dotenv(find_dotenv())

    main()
