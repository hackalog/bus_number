from functools import partial
import importlib
import sys
import json
import os
import pathlib
from sklearn.preprocessing import StandardScaler, MinMaxScaler

from .dset import Dataset, RawDataset
from ..logging import logger
from ..utils import load_json, save_json

__all__ = [
    'add_dataset',
    'available_datasets',
    'load_dataset',
]

_MODULE = sys.modules[__name__]
_MODULE_DIR = pathlib.Path(os.path.dirname(os.path.abspath(__file__)))

def add_dataset(rawds):
    """Add a raw dataset to the list of available datasets"""

    rawds_list, rds_file_fq = available_datasets(keys_only=False)
    rawds_list[rawds.name] = rawds.to_dict()
    save_json(rds_file_fq, rawds_list)

def load_dataset(dataset_name,
                 cache_path=None,
                 fetch_path=None,
                 force=False,
                 unpack_path=None,
                 **kwargs):

    '''Loads a Dataset object by name.

    Dataset will be cached after creation. Subsequent calls with matching call
    signature will return this cached object.

    Parameters
    ----------
    dataset_name:
        Name of dataset to load. see `available_datasets()` for the current list
        be returned (if available)
    cache_path: path
        Directory to search for Dataset cache files
    fetch_path: path
        Directory to download raw files into
    force: boolean
        If True, always regenerate the dataset. If false, a cached result can
    unpack_path: path
        Directory to unpack raw files into

    Remaining keywords arguments are passed to the RawDataset's `process()` method
    '''
    dataset_list, _ = available_datasets(keys_only=False)
    if dataset_name not in dataset_list:
        raise Exception(f'Unknown Dataset: {dataset_name}')
    raw_ds = RawDataset.from_dict(dataset_list[dataset_name])
    raw_ds.fetch(fetch_path=fetch_path, force=force)
    raw_ds.unpack(unpack_path=unpack_path, force=force)
    ds = raw_ds.process(cache_path=cache_path, force=force, **kwargs)

    return ds

def available_datasets(raw_dataset_file='datasets.json', raw_dataset_path=None, keys_only=True):
    """Returns the list of available datasets.

    Instructions for creating RawDatasets is stored in `datasets.json` by default.

    keys_only: boolean
        if True, return a list of available datasets (default)
        if False, return complete dataset dictionary and filename

    Returns
    -------
    If `keys_only` is True:
        List of available dataset names
    else:
        Tuple (available_dataset_dict, available_dataset_dict_filename)
    """
    if raw_dataset_path is None:
        raw_dataset_path = _MODULE_DIR

    raw_dataset_file_fq = pathlib.Path(raw_dataset_path) / raw_dataset_file

    if not raw_dataset_file_fq.exists():
        raw_dataset_dict = {}
        logger.warning(f"No dataset file found: {raw_dataset_file}")
    else:
        raw_dataset_dict = load_json(raw_dataset_file_fq)

    if keys_only:
        return list(raw_dataset_dict.keys())

    return raw_dataset_dict, raw_dataset_file_fq
