'''
Instructions for joining up parts of a Reproducible Data Science workflow

RawDataset: make raw

Dataset: make data

Models: make train

Experiments: make predict

'''

import logging
import os
import pathlib
import sys

from .utils import load_json, save_json
from .paths import raw_data_path, interim_data_path, processed_data_path, model_path
from .data import Dataset
from .logging import logger

_MODULE = sys.modules[__name__]
_MODULE_DIR = pathlib.Path(os.path.dirname(os.path.abspath(__file__)))

def available_datasets(dataset_path=None, keys_only=True):
    """Get a list of available datasets.

    Parameters
    ----------
    dataset_path: path
        location of saved dataset files
    """
    if dataset_path is None:
        dataset_path = processed_data_path
    else:
        dataset_path = pathlib.Path(dataset_path)

    ds_dict = {}
    for dsfile in dataset_path.glob("*.metadata"):
        ds_stem = str(dsfile.stem)
        ds_meta = Dataset.load(ds_stem, data_path=dataset_path, metadata_only=True)
        ds_dict[ds_stem] = ds_meta

    if keys_only:
        return list(ds_dict.keys())
    return ds_dict


def add_raw_dataset(rawds):
    """Add a raw dataset to the list of available raw datasets"""

    rawds_list, rds_file_fq = available_raw_datasets(keys_only=False)
    rawds_list[rawds.name] = rawds.to_dict()
    save_json(rds_file_fq, rawds_list)

def available_raw_datasets(raw_dataset_file='raw_datasets.json',
                           raw_dataset_path=None, keys_only=True):
    """Returns the list of available datasets.

    Instructions for creating RawDatasets is stored in `raw_datasets.json` by default.

    keys_only: boolean
        if True, return a list of available datasets (default)
        if False, return complete dataset dictionary and filename

    Returns
    -------
    If `keys_only` is True:
        List of available dataset names
    else:
        Tuple (available_raw_dataset_dict, available_raw_dataset_dict_filename)
    """
    if raw_dataset_path is None:
        raw_dataset_path = _MODULE_DIR / 'data'

    raw_dataset_file_fq = pathlib.Path(raw_dataset_path) / raw_dataset_file

    if not raw_dataset_file_fq.exists():
        raw_dataset_dict = {}
        logger.warning(f"No dataset file found: {raw_dataset_file}")
    else:
        raw_dataset_dict = load_json(raw_dataset_file_fq)

    if keys_only:
        return list(raw_dataset_dict.keys())

    return raw_dataset_dict, raw_dataset_file_fq

def get_transformer_list(transformer_path=None, transformer_file=None, include_filename=False):
    """Get the list of transformation pipelines

    Returns
    -------
    If include_filename is True:
        A tuple: (transformer_list, transformer_file_fq)
    else:
        transformer_list

    Parameters
    ----------
    include_filename: boolean
        if True, returns a tuple: (list, filename)
    transformer_path: path. (default: MODULE_DIR/data)
        Location of `transformer_file`
    transformer_file: string, default 'transformer_list.json'
        Name of json file that contains the transformer pipeline
    """
    if transformer_path is None:
        transformer_path = _MODULE_DIR / 'data'
    else:
        transformer_path = pathlib.Path(transformer_path)
    if transformer_file is None:
        transformer_file = 'transformer_list.json'

    transformer_file_fq = transformer_path / transformer_file
    try:
        transformer_list = load_json(transformer_file_fq)
    except FileNotFoundError:
        transformer_list = []

    if include_filename:
        return transformer_list, transformer_file_fq
    return transformer_list

def del_transformer(index, transformer_path=None, transformer_file=None):
    """Delete an entry in the transformer list

    index: index of entry
    transformer_path: path. (default: MODULE_DIR/data)
        Location of `transformer_file`
    transformer_file: string, default 'transformer_list.json'
        Name of json file that contains the transformer pipeline
    """
    transformer_list, transformer_file_fq = get_transformer_list(transformer_path=transformer_path,
                                                                 transformer_file=transformer_file,
                                                                 include_filename=True)

    del(transformer_list[index])
    save_json(transformer_file_fq, transformer_list)

def add_transformer(from_raw=None, raw_dataset_opts=None,
                    input_dataset=None, suppress_output=False, output_dataset=None,
                    transformations=None,
                    transformer_path=None, transformer_file=None):
    """Create and add a dataset transformation pipeline to the workflow.

    Transformer pipelines apply a sequence of transformer functions to a Dataset (or RawDataset),
    producing a new Dataset.

    Parameters
    ----------
    input_dataset: string
        Name of a dataset_dir
        Specifying this option creates a dataset transformation pipeline that begins
        with an existing dataset_dir
    from_raw: string
        Name of a raw dataset.
        Specifying this option creates a dataset transformation pipeline that begins
        starts from a raw dataset with this namew
    output_dataset: string
        Name to use when writing the terminal Dataset object to disk.
    raw_dataset_opts: dict
        Options to use when generating raw dataset
    suppress_output: boolean
        If True, the terminal dataset object is not written to disk.
        This is useful when one of the intervening tranformers handles the writing; e.g. train/test split.
    transformeations: list of tuples
        Squence of transformer functions to apply. tuples consist of:
        (transformer_name, transformer_opts)
    transformer_path: path. (default: MODULE_DIR/data)
        Location of `transformer_file`
    transformer_file: string, default 'transformer_list.json'
        Name of json file that contains the transformer pipeline
    """

    if from_raw is not None and input_dataset is not None:
        raise Exception('Cannot set both `from_raw` and `input_datset`')
    if from_raw is None and raw_dataset_opts is not None:
        raise Exception('Must specify `from_raw` when using `raw_dataset_opts`')

    transformer_list, transformer_file_fq = get_transformer_list(transformer_path=transformer_path,
                                                                 transformer_file=transformer_file,
                                                                 include_filename=True)

    transformer = {}
    if from_raw:
        transformer['raw_dataset_name'] = from_raw
        if output_dataset is None and not suppress_output:
            output_dataset = from_raw
    elif input_dataset:
        transformer['input_dataset'] = input_dataset
    else:
        raise Exception("Must specify one of from `from_raw` or `input_dataset`")

    if raw_dataset_opts:
        transformer['raw_dataset_opts'] = raw_dataset_opts

    if transformations:
        transformer['transformations'] = transformations

    if not suppress_output:
        if output_dataset is None:
            raise Exception("Must specify `output_dataset` (or use `suppress_output`")
        else:
            transformer['output_dataset'] = output_dataset

    transformer_list.append(transformer)
    save_json(transformer_file_fq, transformer_list)
