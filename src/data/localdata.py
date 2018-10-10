__all__ = ['process_mnist']

from ..paths import interim_data_path

import numpy as np

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
