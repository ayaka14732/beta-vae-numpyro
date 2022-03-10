import flax.linen as nn
import jax
import jax.numpy as np
import jax.random as rand
import matplotlib.pyplot as plt
import numpy as onp
import numpyro
from numpyro.contrib.module import flax_module
import numpyro.distributions as dist
from numpyro.infer import SVI, Trace_ELBO
import optax
import pickle
import time

# Dataset

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

data_x = load_chairs()
data_size, dim_feature = data_x.shape

image_size = 128
assert image_size * image_size == dim_feature

# Model

dim_z = 50
batch_size = 128
n_epochs = 80
learning_rate = 0.00008  # MNIST: 0.001
beta = 4

class VAEEncoder(nn.Module):
    @nn.compact
    def __call__(self, x: np.ndarray):
        batch_size, _ = x.shape
        x = x.reshape(batch_size, image_size, image_size, 1)  # (b, 128, 128, 1)
        x = nn.softplus(nn.Conv(16, (4, 4), 1)(x))  # (b, 128, 128, 16)
        x = nn.softplus(nn.Conv(32, (4, 4), 2)(x))  # (b, 64, 64, 32)
        x = nn.softplus(nn.Conv(64, (4, 4), 2)(x))  # (b, 32, 32, 64)
        x = x.reshape(batch_size, -1)  # (b, 32 * 32 * 64)

        z_loc = nn.Dense(dim_z)(x)
        z_std = np.exp(nn.Dense(dim_z)(x))

        return z_loc, z_std

class VAEDecoder(nn.Module):
    @nn.compact
    def __call__(self, z: np.ndarray):
        batch_size, _ = z.shape
        z = nn.softplus(nn.Dense(32 * 32 * 64)(z))
        z = z.reshape(batch_size, 32, 32, 64)  # (b, 32, 32, 64)
        z = nn.softplus(nn.ConvTranspose(32, (4, 4), (2, 2))(z))  # (b, 64, 64, 32)
        z = nn.softplus(nn.ConvTranspose(16, (4, 4))(z))  # (b, 128, 128, 16)
        z = nn.softplus(nn.ConvTranspose(1, (4, 4))(z))  # (b, 128, 128, 1)
        z = z.reshape(batch_size, -1)  # (b, 128 * 128)
        z = nn.sigmoid(nn.Dense(dim_feature)(z))
        return z

encoder_nn = VAEEncoder()
decoder_nn = VAEDecoder()

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

key = rand.PRNGKey(42)

optimizer = optax.adam(learning_rate=learning_rate)
svi = SVI(model, guide, optimizer, Trace_ELBO())

key, subkey = rand.split(key)
sample_batch_idx = rand.permutation(subkey, data_size)[:batch_size]
sample_batch = data_x[sample_batch_idx]

key, subkey = rand.split(key)
svi_state = svi.init(subkey, sample_batch)

num_data = data_size // batch_size

update = jax.jit(svi.update)
evaluate = jax.jit(svi.evaluate)

def train_step(key, svi_state):
    shuffled_idx = rand.permutation(key, data_size)
    for i in range(num_data):
        x = data_x[shuffled_idx[i*batch_size:(i+1)*batch_size]]
        svi_state, _ = update(svi_state, x)
    return svi_state

def test_step(svi_state):
    test_loss = 0.0
    for i in range(num_data):
        x = data_x[i*batch_size:(i+1)*batch_size]
        loss = evaluate(svi_state, x) / batch_size
        test_loss += loss
    test_loss /= num_data
    return test_loss

@jax.jit
def reconstruct(params, x, key):
    params_encoder = params['encoder$params']
    z_mean, z_var = encoder_nn.apply({'params': params_encoder}, x)
    z = dist.Normal(z_mean, z_var).sample(key)
    params_decoder = params['decoder$params']
    x_loc = decoder_nn.apply({'params': params_decoder}, z)
    img = (x * 255.).astype(np.int32).reshape(image_size, image_size)
    img_loc = (x_loc * 255.).astype(np.int32).reshape(image_size, image_size)
    imgs = np.hstack((img, img_loc))
    return imgs

def reconstruct_img(params, key, epoch):
    key, subkey = rand.split(key)
    idx = rand.choice(subkey, data_size)
    x = data_x[idx][None, ...]

    key, subkey = rand.split(key)
    imgs = reconstruct(params, x, subkey)
    plt.imsave(f'.results/epoch{epoch}.png', imgs, cmap='gray')

for epoch in range(1, n_epochs + 1):
    time_start = time.time()

    key, subkey = rand.split(key)
    svi_state = train_step(subkey, svi_state)
    test_loss = test_step(svi_state)

    key, subkey = rand.split(key)
    reconstruct_img(svi.get_params(svi_state), subkey, epoch)

    time_elapsed = time.time() - time_start
    print(f'Epoch {epoch}, loss {test_loss:.2f}, time {time_elapsed:.2f}s')

pickle.dump(svi.get_params(svi_state)['encoder$params'], 'params_chair_encoder.pickle')
pickle.dump(svi.get_params(svi_state)['decoder$params'], 'params_chair_decoder.pickle')
