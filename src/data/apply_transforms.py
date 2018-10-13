# -*- coding: utf-8 -*-
import os
from pathlib import Path
from dotenv import find_dotenv, load_dotenv
from ..utils import save_json, load_json
import click

import logging
from ..logging import logger
from ..paths import processed_data_path, src_module_dir
from .transformers import available_transformers
from .datasets import RawDataset, Dataset
from ..workflow import get_transformer_list, available_raw_datasets

@click.command()
@click.argument('transformer_file')
@click.option('--output_dir', '-o', nargs=1, type=str)
@click.option('--input_dir', '-i', nargs=1, type=str)
@click.option('--hash-type', '-H', type=click.Choice(['md5', 'sha1']), default='sha1')
def main(transformer_file, output_dir=None, input_dir=None, *, hash_type):
    logger.info(f'Transforming datasets from {transformer_file}')
    if output_dir is None:
        output_dir = processed_data_path
    else:
        output_dir = Path(output_dir)
    if input_dir is None:
        input_dir = src_module_dir / 'data'
    else:
        input_dir = Path(input_dir)


    raw_datasets = available_raw_datasets()
    transformers = available_transformers()
    os.makedirs(output_dir, exist_ok=True)

    transformer_list = get_transformer_list(transformer_path=input_dir, transformer_file=transformer_file)

    for tdict in transformer_list:
        raw_dataset_opts = tdict.get('raw_dataset_opts', {})
        raw_dataset_name = tdict.get('raw_dataset_name', None)
        output_dataset = tdict.get('output_dataset', None)
        input_dataset = tdict.get('input_dataset', None)
        transformations = tdict.get('transformations', [])
        if raw_dataset_name is not None:
            if raw_dataset_name not in raw_datasets:
                raise Exception(f"Unknown RawDataset: {raw_dataset_name}")
            logger.debug(f"Creating Dataset from Raw: {raw_dataset_name} with opts {raw_dataset_opts}")
            rds = RawDataset.from_name(raw_dataset_name)
            ds = rds.process(**raw_dataset_opts)
        else:
            logger.debug("Loading Dataset: {input_dataset}")
            ds = Dataset.load(input_dataset)

        for tname, topts in transformations:
            tfunc = transformers[tname]
            logger.debug(f"Applying {tname} to {ds.name} with opts {topts}")
            ds = tfunc(ds, **topts)

        if output_dataset is not None:
            logger.info(f"Writing transformed Dataset: {output_dataset}")
            ds.name = output_dataset
            ds.dump(dump_path=output_dir)

if __name__ == '__main__':

    # not used in this stub but often useful for finding various files
    project_dir = Path(__file__).resolve().parents[2]

    # find .env automagically by walking up directories until it's found, then
    # load up the .env entries as environment variables
    load_dotenv(find_dotenv())

    main()
