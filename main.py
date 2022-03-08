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

# Dataset

def load_mnist() -> np.ndarray:
    train_set, test_set = load_dataset('mnist', split=('train', 'test'))

    train_x = onp.asarray([onp.asarray(x, dtype=onp.float32).reshape(-1) for x in train_set['image']]) / 255.
    test_x = onp.asarray([onp.asarray(x, dtype=onp.float32).reshape(-1) for x in test_set['image']]) / 255.

    train_x = np.asarray(train_x, dtype=np.bfloat16)
    test_x = np.asarray(test_x, dtype=np.bfloat16)

    return train_x, test_x

def load_chairs() -> np.ndarray:
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

    train_x = np.array(train_x, dtype=np.bfloat16)
    test_x = np.array(test_x, dtype=np.bfloat16)

    return train_x, test_x

train_x, test_x = load_chairs()
train_size, dim_feature = train_x.shape
test_size, _ =  test_x.shape

image_size = 128
assert image_size * image_size == dim_feature

# Model

dim_hidden = 400
dim_z = 50
batch_size = 200
num_epochs = 15
learning_rate = 0.0001  # MNIST: 0.001
beta = 0.5

class VAEEncoder(nn.Module):
    dim_hidden: int
    dim_z: int

    @nn.compact
    def __call__(self, x: np.ndarray):
        x = nn.Dense(self.dim_hidden)(x)
        x = nn.softplus(x)

        z_loc = nn.Dense(self.dim_z)(x)
        z_std = np.exp(nn.Dense(self.dim_z)(x))

        return z_loc, z_std

class VAEDecoder(nn.Module):
    dim_hidden: int
    dim_feature: int

    @nn.compact
    def __call__(self, z: np.ndarray):
        z = nn.Dense(self.dim_hidden)(z)
        z = nn.softplus(z)
        z = nn.Dense(self.dim_feature)(z)
        x = nn.sigmoid(z)
        return x

encoder_nn = VAEEncoder(dim_hidden, dim_z)
decoder_nn = VAEDecoder(dim_hidden, dim_feature)

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

optimizer = optax.adabelief(learning_rate=learning_rate)
svi = SVI(model, guide, optimizer, Trace_ELBO())

key, subkey = rand.split(key)
sample_batch_idx = rand.permutation(subkey, train_size)[:batch_size]
sample_batch = train_x[sample_batch_idx]

key, subkey = rand.split(key)
svi_state = svi.init(subkey, sample_batch)

num_train = train_size // batch_size
num_test = test_size // batch_size

def train_step(svi_state):
    f = jax.jit(svi.update)
    for i in range(num_train):
        x = train_x[i*batch_size:(i+1)*batch_size]
        svi_state, _ = f(svi_state, x)
    return svi_state

def test_step(svi_state):
    total_loss = 0.0
    f = jax.jit(svi.evaluate)
    for i in range(num_test):
        x = test_x[i*batch_size:(i+1)*batch_size]
        loss = f(svi_state, x) / batch_size
        total_loss += loss
    loss /= num_test
    return loss

def reconstruct_img(params, key, epoch):
    key, subkey = rand.split(key)
    idx = rand.choice(subkey, test_size)
    x = test_x[idx]
    z_mean, z_var = encoder_nn.apply({'params': params['encoder$params']}, x)
    key, subkey = rand.split(key)
    z = dist.Normal(z_mean, z_var).sample(subkey)
    x_loc = decoder_nn.apply({'params': params['decoder$params']}, z)
    img = (x * 255.).astype(np.int32).reshape(image_size, image_size)
    img_loc = (x_loc * 255.).astype(np.int32).reshape(image_size, image_size)
    plt.imsave(f'.results/orig_epoch{epoch}.png', img, cmap='gray')
    plt.imsave(f'.results/reco_epoch{epoch}.png', img_loc, cmap='gray')

for i in range(1, num_epochs + 1):
    time_start = time.time()

    svi_state = train_step(svi_state)
    test_loss = test_step(svi_state)

    key, subkey = rand.split(key)
    reconstruct_img(svi.get_params(svi_state), subkey, i)

    time_elapsed = time.time() - time_start
    print(f'Epoch {i}, loss {test_loss:.2f}, time {time_elapsed:.2f}s')
