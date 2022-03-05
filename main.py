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

train_size = 60000
test_size = 10000
dim_feature = 28 * 28

dim_hidden = 400
dim_z = 60
batch_size = 200
num_epochs = 40
learning_rate = 0.001
beta = 0.05

def load_mnist(split: str) -> onp.ndarray:
    dataset = load_dataset('mnist', split=split)
    x = onp.asarray([onp.asarray(x, dtype=onp.float32).reshape(-1) for x in dataset['image']]) / 255.
    return x

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

def main():
    key = rand.PRNGKey(42)

    optimizer = optax.adabelief(learning_rate=learning_rate)
    svi = SVI(model, guide, optimizer, Trace_ELBO())

    train_x = load_mnist(split='train')
    test_x = load_mnist(split='test')
    assert train_x.shape == (train_size, dim_feature)
    assert test_x.shape == (test_size, dim_feature)

    key, subkey = rand.split(key)
    sample_batch_idx = rand.permutation(subkey, train_size)[:batch_size]
    sample_batch = train_x[sample_batch_idx]

    key, subkey = rand.split(key)
    svi_state = svi.init(subkey, sample_batch)

    num_train = train_size // batch_size
    num_test = test_size // batch_size

    @jax.jit
    def train_step(svi_state):
        def f(i, svi_state):
            x = jax.lax.dynamic_slice(train_x, (i, 0), (batch_size, dim_feature))
            svi_state, _ = svi.update(svi_state, x)
            return svi_state
        svi_state = jax.lax.fori_loop(0, num_train, f, svi_state)
        return svi_state

    @jax.jit
    def test_step(svi_state):
        def f(i, total_loss):
            x = jax.lax.dynamic_slice(test_x, (i, 0), (batch_size, dim_feature))
            loss = svi.evaluate(svi_state, x) / batch_size
            return total_loss + loss
        loss = jax.lax.fori_loop(0, num_test, f, 0.0)
        loss = loss / num_test
        return loss

    def reconstruct_img(params, key, epoch):
        key, subkey = rand.split(key)
        idx = rand.choice(subkey, test_size)
        x = test_x[idx]
        z_mean, z_var = encoder_nn.apply({'params': params['encoder$params']}, x)
        key, subkey = rand.split(key)
        z = dist.Normal(z_mean, z_var).sample(subkey)
        x_loc = decoder_nn.apply({'params': params['decoder$params']}, z)
        img = (x * 255.).astype(np.int32).reshape(28, 28)
        img_loc = (x_loc * 255.).astype(np.int32).reshape(28, 28)
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

if __name__ == '__main__':
    main()
