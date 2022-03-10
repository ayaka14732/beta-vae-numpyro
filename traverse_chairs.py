# %%
import jax
import jax.numpy as np
import jax.random as rand
import matplotlib.pyplot as plt
import numpyro.distributions as dist
import pickle

from lib import load_dataset, VAEEncoder, VAEDecoder

jax.config.update('jax_platform_name', 'cpu')

data_x = load_dataset(dataset='chairs')

data_size, dim_feature = data_x.shape

image_size = 128
assert dim_feature == image_size * image_size

dim_z = 50

key = rand.PRNGKey(42)

encoder_nn = VAEEncoder(image_size, dim_z)
decoder_nn = VAEDecoder(image_size)

with open('params_chairs_encoder.pickle', 'rb') as f:
    params_encoder = pickle.load(f)

with open('params_chairs_decoder.pickle', 'rb') as f:
    params_decoder = pickle.load(f)

# %%
def reconstruct(x, delta=2., n=8, target_dims=(0, 1)):
    # img = (x * 255.).astype(np.int32).reshape(image_size, image_size)

    z_mean, z_var = encoder_nn.apply({'params': params_encoder}, x)
    z = dist.Normal(z_mean, z_var).sample(key)

    delta_axis = slice(-delta, delta, n * 1j)
    n_axis = 2
    assert len(target_dims) == n_axis

    delta_grid = np.mgrid[(delta_axis,) * n_axis].transpose((1, 2, 0))
    z = z[0].tile((n, n, 1)).at[:, :, target_dims].add(delta_grid)

    z = z.reshape(n * n, dim_z)

    x_loc = decoder_nn.apply({'params': params_decoder}, z)
    img_loc = (x_loc * 255.).astype(np.int32)
    
    img_loc = img_loc.reshape(n, n, image_size, image_size)
    imgs = np.hstack(np.hstack(img_loc))
    return imgs

# %%
# key, subkey = rand.split(key)
# idx = rand.choice(subkey, data_size)
idx = 81941
x = data_x[idx][None, ...]

key, subkey = rand.split(key)
imgs = reconstruct(x, delta=2., target_dims=(0, 15))
plt.imsave('1.png', imgs, cmap='gray')

# %%
