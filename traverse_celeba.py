# %%
import jax
import jax.numpy as np
import jax.random as rand
import matplotlib.pyplot as plt
import numpyro.distributions as dist
import pickle

from lib import load_dataset, VAEEncoder, VAEDecoder

jax.config.update('jax_platform_name', 'cpu')

dataset = 'celeba'  # 'chairs'
data_x = load_dataset(dataset=dataset)
data_size, *image_shape = data_x.shape
image_size = image_shape[0]

dim_z = 56
beta = 4

key = rand.PRNGKey(42)

encoder_nn = VAEEncoder(dim_z)
decoder_nn = VAEDecoder()

with open(f'params_{dataset}_{beta}_encoder.pickle', 'rb') as f:
    params_encoder = pickle.load(f)

with open(f'params_{dataset}_{beta}_decoder.pickle', 'rb') as f:
    params_decoder = pickle.load(f)

# %%
def traverse(x, delta=2., n=8, target_dims=(0, 1)):
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
    img_loc = (x_loc * 255.).astype(np.uint8)
    
    img_loc = img_loc.reshape(n, n, *image_shape)
    imgs = np.hstack(np.hstack(img_loc))
    return imgs

# %%
# key, subkey = rand.split(key)
# idx = rand.choice(subkey, data_size)
idx = 255  # 228, 230, 244, 252
x = data_x[idx][None, ...]

key, subkey = rand.split(key)
imgs = traverse(x, delta=3., n=5, target_dims=(3, 24))
plt.axis('off')
plt.imshow(imgs)
plt.show()
plt.imsave(f'traverse_{dataset}_{beta}.png', imgs)

# %%
