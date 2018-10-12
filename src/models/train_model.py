# -*- coding: utf-8 -*-
import click
import json
import os
from pathlib import Path
from dotenv import find_dotenv, load_dotenv

from ..logging import logger
from ..utils import save_json
from ..paths import model_path, trained_model_path
from .train import train_model, save_model
from ..data.datasets import available_datasets
from .algorithms import available_algorithms


@click.command()
@click.argument('model_list')
@click.option('--output_file', '-o', nargs=1, type=str,
              default='trained_models.json')
@click.option('--hash-type', '-H', type=click.Choice(['md5', 'sha1']),
              default='sha1')
def main(model_list, *, output_file, hash_type):
    """Trains models speficied in the supplied `model_list` file

    output is a dictionary of trained model metadata keyed by
    model_key := {algorithm}_{dataset}_{run_number}, where:

    dataset:
        name of dataset to use
    algorithm:
        name of algorithm (estimator) to run on the dataset
    run_number:
        Arbitrary integer.

    The combination of these 3 things must be unique.

    trained models are written to `trained_model_path`.

    For every model, we write:

    {model_key}.model:
        the trained model
    {model_key}.metadata:
        Metadata for this model

    Parameters
    ----------
    model_list: filename
        json file specifying list of options dictionaries to be passed to
        `train_model`
    output_file: str
        name of json file to write metadata to
    hash_name: {'sha1', 'md5'}
        type of hash to use for caching of python objects


    """
    logger.info(f'Building models from {model_list}')

    os.makedirs(trained_model_path, exist_ok=True)

    with open(model_path / model_list) as f:
        training_dicts = json.load(f)

    dataset_list = available_datasets()
    algorithm_list = available_algorithms()

    metadata_dict = {}  # Used to ensure uniqueness of keys
    for td in training_dicts:
        ds_name = td.get('dataset', None)
        assert ds_name in dataset_list, f'Unknown Dataset: {ds_name}'

        alg_name = td.get('algorithm', None)
        assert alg_name in algorithm_list, f'Unknown Algorithm: {alg_name}'

        run_number = td.get('run_number', 0)
        model_key = f"{td['algorithm']}_{td['dataset']}_{run_number}"
        if model_key in metadata_dict:
            raise Exception("{id_base} already exists. Give a unique " +
                            "`run_number` to avoid collisions.")
        else:
            td['run_number'] = run_number
            metadata_dict[model_key] = td

        saved_meta = {}
        for model_key, td in metadata_dict.items():
            logger.debug(f'Creating model for {model_key}')
            trained_model, added_metadata = train_model(hash_type=hash_type,
                                                        **td)
            # replace specified params with full set of params used
            td['algorithm_params'] = dict(trained_model.get_params())
            new_metdata = {**td, **added_metadata}
            saved_meta[model_key] = save_model(model_name=model_key,
                                               model=trained_model,
                                               metadata=new_metdata)

    logger.debug(f"output dir: {model_path}")
    logger.debug(f"output filename: {output_file}")
    save_json(model_path / output_file, saved_meta)
    logger.info("Training complete! Saved metdata to " +
                f"{model_path / output_file}")

if __name__ == '__main__':

    # not used in this stub but often useful for finding various files
    project_dir = Path(__file__).resolve().parents[2]

    # find .env automagically by walking up directories until it's found, then
    # load up the .env entries as environment variables
    load_dotenv(find_dotenv())

    main()
