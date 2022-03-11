#!/bin/sh

# celeba, shape: (202599, 218, 178, 3)
mkdir -p ~/.beta-vae/celeba
kaggle datasets download -d jessicali9530/celeba-dataset -p ~/.beta-vae/celeba
unzip -n ~/.beta-vae/celeba/celeba-dataset.zip -d ~/.beta-vae/celeba/dataset

# chairs, shape: (86366, 128, 128)
mkdir -p ~/.beta-vae/chairs
wget -nc -P ~/.beta-vae/chairs https://www.di.ens.fr/willow/research/seeing3Dchairs/data/rendered_chairs.tar
tar --skip-old-files -xvf ~/.beta-vae/chairs/rendered_chairs.tar -C ~/.beta-vae/chairs

# faces
# See https://faces.dmi.unibas.ch/bfm/bfm2019.html
