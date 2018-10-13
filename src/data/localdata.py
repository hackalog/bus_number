"""
Custom dataset processing/generation functions should be added to this file
"""
import random

import numpy as np
import pandas as pd

from src.data.utils import read_space_delimited, normalize_labels, reservoir_sample
from src.paths import interim_data_path

__all__ = [
    'process_lvq_pak',
    'process_mnist',
    'process_yelp',
]

def process_lvq_pak(dataset_name='lvq-pak', kind='all', numeric_labels=True, metadata=None):
    """
    kind: {'test', 'train', 'all'}, default 'all'
    numeric_labels: boolean (default: True)
        if set, target is a vector of integers, and label_map is created in the metadata
        to reflect the mapping to the string targets
    """

    untar_dir = interim_data_path / dataset_name
    unpack_dir = untar_dir / 'lvq_pak-3.1'

    if kind == 'train':
        data, target = read_space_delimited(unpack_dir / 'ex1.dat', skiprows=[0,1])
    elif kind == 'test':
        data, target = read_space_delimited(unpack_dir / 'ex2.dat', skiprows=[0])
    elif kind == 'all':
        data1, target1 = read_space_delimited(unpack_dir / 'ex1.dat', skiprows=[0,1])
        data2, target2 = read_space_delimited(unpack_dir / 'ex2.dat', skiprows=[0])
        data = np.vstack((data1, data2))
        target = np.append(target1, target2)
    else:
        raise Exception(f'Unknown kind: {kind}')

    if numeric_labels:
        if metadata is None:
            metadata = {}
        mapped_target, label_map = normalize_labels(target)
        metadata['label_map'] = label_map
        target = mapped_target

    dset_opts = {
        'dataset_name': dataset_name,
        'data': data,
        'target': target,
        'metadata': metadata
    }
    return dset_opts

def process_mnist(dataset_name='mnist', kind='train', metadata=None):
    '''
    Load the MNIST dataset (or a compatible variant; e.g. F-MNIST)

    dataset_name: {'mnist', 'f-mnist'}
        Which variant to load
    kind: {'train', 'test'}
        Dataset comes pre-split into training and test data.
        Indicates which dataset to load
    metadata: dict
        Additional metadata fields will be added to this dict.
        'kind': value of `kind` used to generate a subset of the data
    '''
    if metadata is None:
        metadata = {}

    if kind == 'test':
        kind = 't10k'

    label_path = interim_data_path / dataset_name / f"{kind}-labels-idx1-ubyte"
    with open(label_path, 'rb') as fd:
        target = np.frombuffer(fd.read(), dtype=np.uint8, offset=8)
    dataset_path = interim_data_path / dataset_name / f"{kind}-images-idx3-ubyte"
    with open(dataset_path, 'rb') as fd:
        data = np.frombuffer(fd.read(), dtype=np.uint8,
                                       offset=16).reshape(len(target), 784)
    metadata['subset'] = kind

    dset_opts = {
        'dataset_name': dataset_name,
        'data': data,
        'target': target,
        'metadata': metadata,
    }
    return dset_opts

def yelp_json_to_df(filename):
    """Convert a yelp JSON file to a dataframe

    Yelp data consists of one json-object per line.

    Parameters
    ----------

    Returns
    -------
    dataframe corresponding to the file
    """
    with open(filename, 'r') as f:
        data = f.readlines()

    # remove the trailing CR from each line
    data = map(lambda x: x.rstrip(), data)

    # Pack it up as a json list
    data_json_str = "[" + ','.join(data) + "]"

    # and load it into pandas
    df = pd.read_json(data_json_str)
    return df

def process_yelp(dataset_name='yelp', metadata=None, num_reviews=100000, filename=None, random_seed=None):
    """Convert raw yelp data to a Dataset Options dictionary

    Parameters
    ----------
    num_reviews: int or None
        if set, randomly sample this many reviews from the yelp data
    random_seed: int
        Set for reproducible randomness
    """
    if metadata is None:
        metadata = {}
    if filename is None:
        filename = interim_data_path / 'yelp' / 'yelp_academic_dataset_review.json'
    else:
        filename = pathlib.Path(filename)

    if random_seed is not None:
        metadata['random_seed'] = random_seed
        random.seed(random_seed)

    if num_reviews is None:
        df = yelp_json_to_df(filename)
    else:
        sample = reservoir_sample(filename, n_samples=num_reviews, random_seed=random_seed)
        data_json_str = "[" + ','.join(sample) + "]"
        df = pd.read_json(data_json_str)
        metadata['num_reviews'] = num_reviews

    data = np.array(df.text)

    target = None
    dset_opts = {
        'dataset_name': dataset_name,
        'data': data,
        'target': target,
        'metadata': metadata
    }
    return dset_opts

__all__ += ['process_fremont_bike']
def process_fremont_bike(dataset_name="fremont_bike", metadata=None):
    """ """
    if metadata is None:
        metadata = {}
    data = pd.read_csv(interim_data_path / dataset_name / 'fremont.csv', index_col='Date')

    try:
        data.index = pd.to_datetime(data.index, format='%m/%d/%Y %I:%M:%S %p')
    except TypeError:
        data.index = pd.to_datetime(data.index)
    data.columns = ['West', 'East']
    data['Total'] = data['West'] + data['East']

    return {
        "dataset_name":dataset_name,
        "metadata": metadata,
        "data":data,
        "target":None
    }
