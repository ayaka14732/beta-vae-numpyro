#!/bin/sh

# celeba
mkdir -p ~/.beta-vae/celeba
kaggle datasets download -d jessicali9530/celeba-dataset -p ~/.beta-vae/celeba
unzip -n ~/.beta-vae/celeba/celeba-dataset.zip -d ~/.beta-vae/celeba/dataset  # shape: (202599, 218, 178, 3)

# chairs
mkdir -p ~/.beta-vae/chair
wget -nc -P ~/.beta-vae/chair https://www.di.ens.fr/willow/research/seeing3Dchairs/data/rendered_chairs.tar
tar --skip-old-files -xvf ~/.beta-vae/chair/rendered_chairs.tar -C ~/.beta-vae/chair

# faces
# See https://faces.dmi.unibas.ch/bfm/bfm2019.html
