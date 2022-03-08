#%%
from datasets import load_dataset
import flax.linen as nn
import jax
import jax.numpy as np
import jax.random as rand
import numpy as onp
import numpyro
from numpyro.contrib.module import flax_module
import numpyro.distributions as dist
from numpyro.infer import SVI, Trace_ELBO
import optax
import matplotlib.pyplot as plt
import time

#%%
# Dataset

def load_mnist() -> onp.ndarray:
    train_set, test_set = load_dataset('mnist', split=('train', 'test'))

    train_x = onp.asarray([onp.asarray(x, dtype=onp.float32).reshape(-1) for x in train_set['image']]) / 255.
    test_x = onp.asarray([onp.asarray(x, dtype=onp.float32).reshape(-1) for x in test_set['image']]) / 255.

    return train_x, test_x

def load_chairs() -> onp.ndarray:
    from os.path import expanduser, join

    datafile = join(expanduser('~'), '.beta-vae/chair/chairs.npy')
    x = onp.load(datafile)

    data_size, _, _ = x.shape

    train_size = int(0.8 * data_size)
    test_size = data_size - train_size
    train_x = x[:train_size]
    test_x = x[train_size:]

    train_x = train_x.reshape(train_size, -1)
    test_x = test_x.reshape(test_size, -1)

    return train_x, test_x

train_x, test_x = load_chairs()
train_size, dim_feature = train_x.shape
test_size, _ =  test_x.shape

image_size = 128
assert image_size * image_size == dim_feature

#%%
# Model

dim_z = 50
batch_size = 50
n_epochs = 15
learning_rate = 0.0002  # MNIST: 0.001
beta = 0.5

class VAEEncoder(nn.Module):
    dim_z: int

    @nn.compact
    def __call__(self, x: np.ndarray):
        batch_size, dim_feature = x.shape
        assert dim_feature == image_size * image_size
        x = x.reshape(batch_size, image_size, image_size, 1)

        x = nn.gelu(nn.Conv(8, (3, 3))(x))
        x = nn.gelu(nn.Conv(8, (3, 3))(x))
        x = x.reshape(batch_size, -1)

        z_loc = nn.Dense(self.dim_z)(x)
        z_std = np.exp(nn.Dense(self.dim_z)(x))

        return z_loc, z_std

class VAEDecoder(nn.Module):
    dim_feature: int

    @nn.compact
    def __call__(self, z: np.ndarray):
        batch_size, _ = z.shape
        z = nn.gelu(nn.Dense(image_size * image_size * 8)(z))
        z = z.reshape(batch_size, image_size, image_size, 8)

        z = nn.gelu(nn.ConvTranspose(8, (3, 3))(z))
        z = nn.gelu(nn.ConvTranspose(1, (3, 3))(z))

        z = z.reshape(batch_size, -1)
        z = nn.Dense(self.dim_feature)(z)
        x = nn.sigmoid(z)
        return x

encoder_nn = VAEEncoder(dim_z)
decoder_nn = VAEDecoder(dim_feature)

def model(x: np.ndarray):
    decoder = flax_module('decoder', decoder_nn, input_shape=(batch_size, dim_z))
    with numpyro.plate('batch', batch_size):
        with numpyro.handlers.scale(scale=beta):
            z = numpyro.sample('z', dist.Normal(0, 1).expand([dim_z]).to_event(1))
        img_loc = decoder(z)
        return numpyro.sample('obs', dist.Bernoulli(img_loc).to_event(1), obs=x)

def guide(x: np.ndarray):
    encoder = flax_module('encoder', encoder_nn, input_shape=(batch_size, dim_feature))
    z_loc, z_std = encoder(x)
    with numpyro.plate('batch', batch_size):
        with numpyro.handlers.scale(scale=beta):
            return numpyro.sample('z', dist.Normal(z_loc, z_std).to_event(1))

#%%
key = rand.PRNGKey(42)

optimizer = optax.adabelief(learning_rate=learning_rate)
svi = SVI(model, guide, optimizer, Trace_ELBO())

key, subkey = rand.split(key)
sample_batch_idx = rand.permutation(subkey, train_size)[:batch_size]
sample_batch = train_x[sample_batch_idx]

key, subkey = rand.split(key)
svi_state = svi.init(subkey, sample_batch)

num_train = train_size // batch_size
num_test = test_size // batch_size

#%%
update = jax.jit(svi.update)

evaluate = jax.jit(svi.evaluate)

@jax.jit
def reconstruct(params, x, key):
    params_encoder = params['encoder$params']
    z_mean, z_var = encoder_nn.apply({'params': params_encoder}, x)
    z = dist.Normal(z_mean, z_var).sample(key)
    params_decoder = params['decoder$params']
    x_loc = decoder_nn.apply({'params': params_decoder}, z)
    img = (x * 255.).astype(np.int32).reshape(image_size, image_size)
    img_loc = (x_loc * 255.).astype(np.int32).reshape(image_size, image_size)
    return img, img_loc

def reconstruct_img(params, key, epoch):
    key, subkey = rand.split(key)
    idx = rand.choice(subkey, test_size)
    x = test_x[idx][None, ...]

    key, subkey = rand.split(key)
    img, img_loc = reconstruct(params, x, subkey)

    plt.imsave(f'.results/orig_epoch{epoch}.png', img, cmap='gray')
    plt.imsave(f'.results/reco_epoch{epoch}.png', img_loc, cmap='gray')

for epoch in range(1, n_epochs + 1):
    time_start = time.time()

    # train step
    for i in range(num_train):
        x = np.asarray(train_x[i*batch_size:(i+1)*batch_size], dtype=np.bfloat16)
        svi_state, _ = update(svi_state, x)

    # test step
    test_loss = 0.0
    for i in range(num_test):
        x = np.asarray(test_x[i*batch_size:(i+1)*batch_size], dtype=np.bfloat16)
        loss = evaluate(svi_state, x) / batch_size
        test_loss += loss
    test_loss /= num_test

    # reconstruct
    key, subkey = rand.split(key)
    reconstruct_img(svi.get_params(svi_state), subkey, epoch)

    time_elapsed = time.time() - time_start
    print(f'Epoch {epoch}, loss {test_loss:.2f}, time {time_elapsed:.2f}s')

# %%
