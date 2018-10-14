import pathlib
import pandas as pd

from ..logging import logger
from ..data import Dataset
from ..paths import model_output_path, trained_model_path
from ..workflow import available_predictions
from ..models.model_list import load_model

from sklearn.metrics import accuracy_score

def available_summarizers():
    _SUMMARIZERS = {
        'supervised_score_df': supervised_score_df
    }
    return _SUMMARIZERS

def available_scorers():
    _SCORERS = {
        'accuracy_score': accuracy_score
    }
    return _SCORERS

def run_summarizer(summarizer_name=None, summarizer_params=None):
    '''
    Run a summarizer on the given params.
    '''
    s = available_summarizers()[summarizer_name](**summarizer_params)
    return s

def save_df_summary(df, meta, summary_path):
    name = meta['summarizer_name']
    df.to_csv(summary_path / name)
    return name + '.csv'


def supervised_score_df(model_dir=None,
                        predictions_dir=None,
                        predictions_list=None,
                        score_list=['accuracy_score']):
    '''
    Test predictions against real labels using the scorers given in the score_list.

    Parameters
    ---------
    model_dir:
        path to the trained models
    predictions_dir:
        path to the prediction outputs
    predictions_list:
        (optional) list of predictions to compare. Should be a subset
        of the available_predictions. If None, use all available_predictions.
    score_list:
        List of scorers to use when comparing predicted output vs. real labels

    Returns
    -------
    df
        A dataframe containing the summary results.
    '''
    if predictions_dir is None:
        predictions_dir = model_output_path
    else:
        predictions_dir = pathlib.Path(predictions_dir)

    if model_dir is None:
        model_dir = trained_model_path
    else:
        model_dir = pathlib.Path(model_dir)

    if predictions_list is None:
        predictions = available_predictions(keys_only=False, models_dir=predictions_dir)
    else:
        predictions = predictions_list

    logger.debug(f"Number of predictions to test:{len(predictions)}")

    score_df = pd.DataFrame(columns=['score_name', 'algorithm_name', 'dataset_name',
                                     'model_name', 'run_number'])
    for current_scorer_name in score_list:
        current_scorer = available_scorers()[current_scorer_name]

        score_dict = {}
        score_dict['score_name'] = current_scorer_name
        for key in predictions.keys():
            prediction = predictions[key]
            exp = prediction['experiment']
            pred_ds = Dataset.load(prediction['dataset_name'],
                                   data_path=predictions_dir)

            ds_name = exp['dataset_name']
            ds = Dataset.load(ds_name)
            score_dict['dataset_name'] = ds_name

            score_dict['score'] = current_scorer(ds.target, pred_ds.data)
            logger.info(model_dir)
            model_metadata = load_model(model_name=exp['model_name'],
                                        metadata_only=True,
                                        model_path=model_dir)
            score_dict['algorithm_name'] = model_metadata['algorithm_name']
            score_dict['model_name'] = exp['model_name']
            score_dict['run_number'] = exp['run_number']
            new_score_df = pd.DataFrame(score_dict, index=[0])
            score_df = score_df.append(new_score_df, sort=True)
    return score_df
