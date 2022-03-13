# NumPyro implementation of _β_-VAE

## Run

Prepare data:

```sh
./prepare_data.sh
python preprocess_chairs.py
```

Train:

```sh
python train.py
```

Latent space traversal:

```sh
python traverse_chairs.py
```

## Sample results

![](traverse_celeba_4_1530_3_28.png)

![](traverse_celeba_4_1548_3_28.png)

![](traverse_celeba_4_1594_18_44.png)

![](traverse_celeba_4_1604_40_28.png)
