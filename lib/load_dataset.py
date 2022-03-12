import numpy as onp

def load_mnist() -> onp.ndarray:
    from datasets import load_dataset
    from itertools import chain
    train_set, test_set = load_dataset('mnist', split=('train', 'test'))
    data_x = onp.asarray([onp.asarray(x, dtype=onp.float32).reshape(-1) for x in chain(train_set['image'], test_set['image'])]) / 255.
    return data_x

def load_chairs() -> onp.ndarray:
    from os.path import expanduser, join

    datafile = join(expanduser('~'), '.beta-vae/chairs/chairs.npy')
    data_x = onp.load(datafile)

    return data_x

def load_celeba() -> onp.ndarray:
    from glob import glob
    from os.path import expanduser, join
    from PIL import Image

    filenames = glob(join(expanduser('~'), '.beta-vae/celeba/dataset/img_align_celeba/img_align_celeba/*.jpg'))
    filenames = filenames[:98304]  # size: 98304*216*176*3*4/1024/1024/1024 = ~41.8G

    # cut (202599, 218, 178, 3) -> (202599, 216, 216, 3) for easier nn.Conv
    data_x = onp.asarray([onp.asarray(Image.open(filename), dtype=onp.float32)[:216, :176] for filename in filenames]) / 255.

    return data_x

def load_dataset(dataset: str) -> onp.ndarray:
    if dataset == 'mnist':
        return load_mnist()
    if dataset == 'chairs':
        return load_chairs()
    if dataset == 'celeba':
        return load_celeba()
    raise ValueError("Unsupported dataset, please select one of: ['mnist', 'chairs', 'celeba'].")
