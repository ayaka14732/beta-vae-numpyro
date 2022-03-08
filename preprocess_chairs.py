# ~/.beta-vae/chair/chairs.npy
# < 30 secs on 96 cpus
# shape: (86366, 128, 128)

from glob import glob
from multiprocessing import Pool
import numpy as onp
from os.path import expanduser, join
from PIL import Image, ImageOps
from tqdm import tqdm

def load_one_file(filename: str) -> onp.ndarray:
    return ImageOps.invert(Image.open(filename)
        .convert('L')
        .crop((128, 128, 558, 558))
        .resize((128, 128)))

def main():
    root = join(expanduser('~'), '.beta-vae/chair')
    filenames = glob(join(root, 'rendered_chairs/**/*.png'), recursive=True)
    with Pool() as p:
        images = tqdm(p.imap(load_one_file, filenames), total=len(filenames))
        x = onp.asarray([onp.asarray(image, dtype=onp.float32) for image in images]) / 255.
    onp.save(join(root, 'chairs.npy'), x)

if __name__ == '__main__':
    main()
