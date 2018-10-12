from .datasets import Dataset
from sklearn.model_selection import train_test_split

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
    train_test_split    train_test_split
    ============        ====================================
    """
    return {
        "train_test_split":train_test_split,
    }

def train_test_split(dset,
                     dump_path=None, dump_metadata=True, force=True, create_dirs=True,
                     **split_opts):
    """Perform a train/test split

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
    new_ds['train'].dump(force=force, dump_path=dump_path, dump_metadata=dump_metadata, create_dirs=create_dirs)

    new_ds['test'].data = X_test
    new_ds['test'].target = y_test
    new_ds['test'].dump(force=force, dump_path=dump_path, dump_metadata=dump_metadata, create_dirs=create_dirs)
