# -*- coding: utf-8 -*-
import click
import json
import os
from pathlib import Path
from dotenv import find_dotenv, load_dotenv

from ..logging import logger
from ..utils import load_json, save_json
from ..paths import reports_path, summary_path
from .analysis import run_summarizer, save_df_summary

@click.command()
@click.argument('summary_list')
@click.option('--output_file', '-o', nargs=1, type=str,
              default='summaries.json')
@click.option('--hash-type', '-H', type=click.Choice(['md5', 'sha1']),
              default='sha1')
def main(summary_list, *, output_file, hash_type):
    """Runs summary analysis speficied in the supplied `summary_list` file.

    The supplied `summary_list` file should be a list of dicts with
    the following key-value pairs.

        summarizer_name: name of a summarizer in available_summarizers
        summarizer_params: dict of params that the summarizer takes

    For every summarizer, we write:

    {summarizer}.csv:
        summary output file
    {summarizer}.metadata:
        Metadata for what went into this summarizer

    Parameters
    ----------
    summarizer_list: filename
        json file specifying list of options dictionaries to be passed to
        `train_model`
    output_file: str
        name of json file to write metadata to
    hash_name: {'sha1', 'md5'}
        type of hash to use for caching of python objects


    """
    logger.debug(f'Running summary analysis from {summary_list}')

    os.makedirs(summary_path, exist_ok=True)

    summary_list = load_json(reports_path / summary_list)

    saved_meta= {}
    for summarizer in summary_list:
        df = run_summarizer(**summarizer)
        filename = save_df_summary(df, summarizer, summary_path=summary_path)
        saved_meta[filename] = summarizer

    if saved_meta:
        save_json(reports_path / output_file, saved_meta)
    logger.info("Summaries complete! Access results via workflow.available_summaries")

if __name__ == '__main__':

    # not used in this stub but often useful for finding various files
    project_dir = Path(__file__).resolve().parents[2]

    # find .env automagically by walking up directories until it's found, then
    # load up the .env entries as environment variables
    load_dotenv(find_dotenv())

    main()
