from .datasets import Dataset
from sklearn.model_selection import train_test_split
from ..logging import logger

def available_transformers():
    """Valid transformation functions

    This function simply returns a dict of known
    tranformer algorithms strings and their corresponding
    function call

    It exists to allow for a description of the mapping for
    each of the valid strings as a docstring

    The valid algorithm names, and the function they map to, are:

    ============        ====================================
    string              Transformer Function
    ============        ====================================
    train_test_split    train_test_split_xform
    ============        ====================================
    """
    return _TRANSFORMERS

def split_dataset_test_train(dset,
                             dump_path=None, dump_metadata=True,
                             force=True, create_dirs=True,
                             **split_opts):
    """Transformer that performs a train/test split.

    This transformer passes `dset` intact, but creates and dumps two new
    datasets as a side effect: {dset.name}_test and {dset.name}_train

    Parameters
    ----------
    dump_metadata: boolean
        If True, also dump a standalone copy of the metadata.
        Useful for checking metadata without reading
        in the (potentially large) dataset itself
    dump_path: path. (default: `processed_data_path`)
        Directory where data will be dumped.
    force: boolean
        If False, raise an exception if any dunp files already exists
        If True, overwrite any existing files
    create_dirs: boolean
        If True, `dump_path` will be created (if necessary)
    **split_opts:
        Remaining options will be passed to `train_test_split`

"""
    new_ds = {}
    for kind in ['train', 'test']:
        dset_name = f"{dset.name}_{kind}"
        dset_meta = {**dset.metadata, 'split':kind, 'split_opts':split_opts}
        new_ds[kind] = Dataset(dataset_name=dset_name, metadata=dset_meta)
    X_train, X_test, y_train, y_test = train_test_split(dset.data, dset.target, **split_opts)

    new_ds['train'].data = X_train
    new_ds['train'].target = y_train
    logger.info(f"Writing Transformed Dataset: {new_ds['train'].name}")
    new_ds['train'].dump(force=force, dump_path=dump_path, dump_metadata=dump_metadata, create_dirs=create_dirs)

    new_ds['test'].data = X_test
    new_ds['test'].target = y_test
    logger.info(f"Writing Transformed Dataset: {new_ds['test'].name}")
    new_ds['test'].dump(force=force, dump_path=dump_path, dump_metadata=dump_metadata, create_dirs=create_dirs)
    return dset

def datetime_pivot(dset, **pivot_opts):
    """Pivot data that is indexed by datatime stamps

    Assumes data is a pandas dataframe, index is a datetime.


    pivot_opts:
        keyword arguments passed to pandas.Dataframe.pivot_table
    """
    pivoted = dset.data.pivot_table(index=dset.data.index.time, columns=dset.data.index.date, **pivot_opts)
    ds_pivot = Dataset(name=f"{dset.name}_pivoted", metadata=dset.metadata, data=pivoted, target=None)
    ds_pivot.metadata['pivot_opts'] = pivot_opts

    return ds_pivot

# A transformer takes a dataset and returns a dataset
def pivot(dset, **pivot_opts):
    """Pivot data stored as a Pandas Dataframe

    pivot_opts:
        keyword arguments passed to pandas.Dataframe.pivot_table
    """
    pivoted = dset.data.pivot_table(**pivot_opts)
    ds_pivot = Dataset(name=f"{dset.name}_pivoted", metadata=dset.metadata, data=pivoted, target=None)
    ds_pivot.metadata['pivot_opts'] = pivot_opts

    return ds_pivot

def index_to_date_time(dset, suffix='dt'):
    """Transformer: Extract a datetime index into Date and Time columns"""
    df = dset.data.copy()
    df['Time']=df.index.time
    df['Date']=df.index.date
    df.reset_index(inplace=True, drop=True)
    new_ds = Dataset(dataset_name=f"{dset.name}_{suffix}", metadata=dset.metadata, data=df)
    return new_ds


_TRANSFORMERS = {
    "datetime_pivot": datetime_pivot,
    "index_to_date_time": index_to_date_time,
    "pivot": pivot,
    "train_test_split": split_dataset_test_train,
}
