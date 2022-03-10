import numpy as onp

def load_mnist() -> onp.ndarray:
    from datasets import load_dataset
    from itertools import chain
    train_set, test_set = load_dataset('mnist', split=('train', 'test'))
    data_x = onp.asarray([onp.asarray(x, dtype=onp.float32).reshape(-1) for x in chain(train_set['image'], test_set['image'])]) / 255.
    return data_x

def load_chairs() -> onp.ndarray:
    from os.path import expanduser, join

    datafile = join(expanduser('~'), '.beta-vae/chair/chairs.npy')
    data_x = onp.load(datafile)

    data_size, _, _ = data_x.shape
    data_x = data_x.reshape(data_size, -1)

    return data_x

def load_dataset(dataset: str) -> onp.ndarray:
    if dataset == 'mnist':
        return load_mnist()
    if dataset == 'chairs':
        return load_chairs()
    raise ValueError("Unsupported dataset, please select one of: ['mnist', 'chairs'].")
