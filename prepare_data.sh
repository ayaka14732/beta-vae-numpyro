#!/bin/sh

# celeba
kaggle datasets download -d jessicali9530/celeba-dataset
unzip celeba-dataset.zip  # shape: (202599, 218, 178, 3)

# chairs
wget https://www.di.ens.fr/willow/research/seeing3Dchairs/data/rendered_chairs.tar

# faces
# See https://faces.dmi.unibas.ch/bfm/bfm2019.html
