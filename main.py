import jax
import jax.numpy as np
import jax.random as rand
import matplotlib.pyplot as plt
import numpyro
from numpyro.contrib.module import flax_module
import numpyro.distributions as dist
from numpyro.infer import SVI, Trace_ELBO
import optax
import pickle
import time

from lib import load_dataset, VAEEncoder, VAEDecoder

# Dataset

data_x = load_dataset(dataset='chairs')
data_x = np.asarray(data_x)  # transfer to default device
data_size, image_size, _ = data_x.shape

# Model

dim_z = 32
batch_size = 64
n_epochs = 100
learning_rate = 0.0001  # MNIST: 0.001
beta = 4

encoder_nn = VAEEncoder(image_size, dim_z)
decoder_nn = VAEDecoder(image_size)

def model(x: np.ndarray):
    decoder = flax_module('decoder', decoder_nn, input_shape=(batch_size, dim_z))
    with numpyro.plate('batch', batch_size):
        with numpyro.handlers.scale(scale=beta):
            z = numpyro.sample('z', dist.Normal(0, 1).expand([dim_z]).to_event(1))
        img_loc = decoder(z)
        return numpyro.sample('obs', dist.Bernoulli(img_loc).to_event(2), obs=x)

def guide(x: np.ndarray):
    encoder = flax_module('encoder', encoder_nn, input_shape=(batch_size, image_size, image_size))
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

def train_step(key, svi_state):
    shuffled_idx = rand.permutation(key, data_size)
    total_loss = 0.0
    for i in range(num_data):
        x = data_x[shuffled_idx[i*batch_size:(i+1)*batch_size]]
        svi_state, loss = update(svi_state, x)
        total_loss += loss / batch_size
    total_loss /= num_data
    return svi_state, total_loss

@jax.jit
def reconstruct(params, x, key):
    img = (x * 255.).astype(np.int32).reshape(image_size, image_size)

    params_encoder = params['encoder$params']
    params_decoder = params['decoder$params']

    z_mean, z_var = encoder_nn.apply({'params': params_encoder}, x)
    z = dist.Normal(z_mean, z_var).sample(key)
    x_loc = decoder_nn.apply({'params': params_decoder}, z)

    img_loc = (x_loc * 255.).astype(np.int32).reshape(image_size, image_size)

    return np.hstack((img, img_loc))

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
    svi_state, total_loss = train_step(subkey, svi_state)

    key, subkey = rand.split(key)
    reconstruct_img(svi.get_params(svi_state), subkey, epoch)

    time_elapsed = time.time() - time_start
    print(f'Epoch {epoch}, loss {total_loss:.2f}, time {time_elapsed:.2f}s')

with open('params_chairs_encoder.pickle', 'wb') as f:
    pickle.dump(svi.get_params(svi_state)['encoder$params'], f)
with open('params_chairs_decoder.pickle', 'wb') as f:
    pickle.dump(svi.get_params(svi_state)['decoder$params'], f)
