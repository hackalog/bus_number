'''
Instructions for joining up parts of a Reproducible Data Science workflow

RawDataset: make raw

Dataset: make data

Models: make train

Experiments: make predict

'''
from .data.transformers import available_transformers
from .models import available_algorithms
from .models.model_list import (get_model_list, add_model, del_model, build_models,
                                available_models)

from .models.predict import (add_prediction, get_prediction_list,
                             pop_prediction, run_predictions, available_predictions)
from .data import (Dataset, RawDataset, available_datasets, available_raw_datasets, add_raw_dataset)
from .data.transform_data import get_transformer_list, add_transformer, del_transformer, apply_transforms

__all__ = [
    'available_datasets',
    'available_raw_datasets',
    'add_raw_dataset',
    'get_transformer_list',
    'apply_transforms',
    'add_transformer',
    'del_transformer',
    'available_transformers',
    'available_algorithms',
    'get_model_list',
    'add_model',
    'del_model',
    'available_models',
    'add_prediction',
    'get_prediction_list',
    'pop_prediction',
    'run_predictions',
    'available_predictions',
]
