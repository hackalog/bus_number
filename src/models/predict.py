import joblib
import json
import os
import pathlib
import time

from ..data import Dataset
from ..logging import logger
from ..paths import model_output_path
from ..utils import record_time_interval
from .train import load_model

__all__ = ['run_model']


def run_model(experiment_info=None,
              file_base=None,
              force=False,
              hash_type='sha1',
              output_path=None,
              run_number=0,
              *,
              dataset_name,
              is_supervised,
              model_name):
    '''Run a model on a dataset (predict/transform)

    Runs an algorithm_object on the dataset and returns a new
    dataset object, tagged with experiment metadata,
    and saves it to disk under `data_path / file_base`.

    Parameters
    ----------
    dataset_name: str, valid dataset name
        Name of a dataset object that will be run through the model
    model_name: str, valid model name
        name of the model that will transform the data
    experiment_info: (str)
        any other information to note about the experiment
        This is used as the output dataset's DESCR text
    output_path: path
        directory to store output files
    file_base: (str, optional) filename base for the output dataset.
        Will also be used as the output `dataset.name`.
    run_number: (int)
        attempt number via the same parameters
    force: (boolean)
        force re-running the algorithm and overwriting any existing data.

    Returns
    -------
    Dataset object emerging from the model,
    with experiment dictionary embedded in metadata
    '''
    if output_path is None:
        output_path = model_output_path
    else:
        output_path = pathlib.Path(output_path)

    if file_base is None:
        file_base = f'{model_name}_exp_{dataset_name}_{run_number}'

    os.makedirs(output_path, exist_ok=True)

    dataset = Dataset.load(dataset_name)

    model, model_meta = load_model(model_name)

    # add experiment metadata
    experiment = {
        'model_name': model_name,
        'dataset_name': dataset_name,
        'run_number': run_number,
        'hash_type': hash_type,
        'data_hash': joblib.hash(dataset.data, hash_name=hash_type),
        'target_hash': joblib.hash(dataset.target, hash_name=hash_type),
        'model_hash': joblib.hash(model, hash_name=hash_type),
    }
    logger.debug(f"Predict: Applying {model_name} on {dataset_name}")
    metadata_fq = output_path / f'{file_base}.metadata'

    if metadata_fq.exists() and force is False:
        cached_metadata = Dataset.load(file_base, data_path=output_path,
                                       metadata_only=True)
        if experiment.items() <= cached_metadata['experiment'].items():
            logger.info("Experiment has already been run. Returning Cached Result")
            return Dataset.load(file_base, data_path=output_path)
        else:
            raise Exception(f'An Experiment with this name exists already, '
                            'but metadata has changed. '
                            'Use `force=True` to overwrite, or change one of '
                            '`run_number` or `file_base`')

    # Either force is True, or we need to rerun the algorithm.
    start_time = time.time()
    if is_supervised:
        exp_data = model.predict(dataset.data)
    else:
        if hasattr(model, 'transform'):
            logger.debug('Transform found. Skipping fit')
            exp_data = model.transform(dataset.data)
        else:
            logger.debug('No Transform found. Running fit_transform')
            exp_data = model.fit_transform(dataset.data)

    end_time = record_time_interval(file_base, start_time)

    experiment['start_time'] = start_time
    experiment['duration'] = end_time - start_time

    new_metadata = dataset.metadata.copy()
    new_metadata['experiment'] = experiment
    new_dataset = Dataset(dataset_name=file_base, data=exp_data,
                          target=dataset.target, metadata=new_metadata,
                          descr_txt=experiment_info)
    new_dataset.dump(file_base=file_base, dump_path=output_path, force=True)
    return new_dataset


def load_prediction(predict_name=None, metadata_only=False, predict_path=None):
    """Load a prediction (or prediction metadata)

    Parameters
    ----------
    metadata_only: boolean
        If True, just return the prediction metadata.
    predict_path:
    predict_name:

    Returns
    -------
    If `metadata_only` is True:

        dict containing predict_metadata

    else:

        The tuple (predict, predict_metadata)
    """
    if predict_name is None:
        raise Exception("predict_name must be specified")
    if predict_path is None:
        predict_path = model_output_path
    else:
        predict_path = pathlib.Path(predict_path)

    fq_predict = predict_path / f'{predict_name}'

    predict = Dataset.load(fq_predict, data_path=predict_path, metadata_only=metadata_only)

    return predict
