# %%
# preprocess geometry Dataset
import jax.numpy as np
from jax.numpy import load
# load dict of arrays
savedir = '/content/train.npz'

dict_data = load(savedir)
# extract the first array
data_x = dict_data['arr_0']
data_x = data_x.reshape(data_x.shape[0], data_x.shape[1]*data_x.shape[2])
data_x.shape

# %%
#preprocess dSprites dataset

import jax.numpy as np
import torch
data = np.load('/content/drive/MyDrive/beta-vae-numpyro/dsprites_ndarray_co1sh3sc6or40x32y32_64x64.npz', encoding='latin1')
data_x = torch.from_numpy(data['imgs'][:128*50]).float().detach().numpy()
data_x = data_x.reshape(data_x.shape[0], data_x.shape[1]*data_x.shape[2])
# print(data_x.shape)
# data_x = data_x.tolist()
# data_x = np.array(data_x)
# print(data_x.shape)

# %%
#beta = 4
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
import numpy as npy
from google.colab.patches import cv2_imshow

from lib import load_dataset, VAEEncoder, VAEDecoder

# geometry Dataset
# with open('train_v2.npy', 'rb') as f:
#     data_x = np.load(f)



data_size, dim_feature = data_x.shape

image_size = 128
assert image_size * image_size == dim_feature

# Model

dim_z = 50
batch_size = 128
n_epochs = 200
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
    # plt.imsave(f'.results/epoch{epoch}.png', imgs, cmap='gray')
    npimgs = npy.array(imgs.tolist()) 
    cv2_imshow(npimgs)

for epoch in range(1, n_epochs + 1):
    time_start = time.time()

    key, subkey = rand.split(key)
    svi_state, total_loss = train_step(subkey, svi_state)

    key, subkey = rand.split(key)
    reconstruct_img(svi.get_params(svi_state), subkey, epoch)

    time_elapsed = time.time() - time_start
    print(f'Epoch {epoch}, loss {total_loss:.2f}, time {time_elapsed:.2f}s')

with open('params_chairs_encoder_beta4.pickle', 'wb') as f:
    pickle.dump(svi.get_params(svi_state)['encoder$params'], f)
with open('params_chairs_decoder_beta4.pickle', 'wb') as f:
    pickle.dump(svi.get_params(svi_state)['decoder$params'], f)

# %%
#beta = 5
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
import numpy as npy
from google.colab.patches import cv2_imshow

from lib import load_dataset, VAEEncoder, VAEDecoder

# geometry Dataset
# with open('train_v2.npy', 'rb') as f:
#     data_x = np.load(f)



data_size, dim_feature = data_x.shape

image_size = 128
assert image_size * image_size == dim_feature

# Model

dim_z = 50
batch_size = 128
n_epochs = 200
learning_rate = 0.00005  # MNIST: 0.001
beta = 5

encoder_nn = VAEEncoder(image_size, dim_z)
decoder_nn = VAEDecoder(image_size)

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
    # plt.imsave(f'.results/epoch{epoch}.png', imgs, cmap='gray')
    npimgs = npy.array(imgs.tolist()) 
    cv2_imshow(npimgs)

for epoch in range(1, n_epochs + 1):
    time_start = time.time()

    key, subkey = rand.split(key)
    svi_state, total_loss = train_step(subkey, svi_state)

    key, subkey = rand.split(key)
    reconstruct_img(svi.get_params(svi_state), subkey, epoch)

    time_elapsed = time.time() - time_start
    print(f'Epoch {epoch}, loss {total_loss:.2f}, time {time_elapsed:.2f}s')

with open('params_chairs_encoder_beta5.pickle', 'wb') as f:
    pickle.dump(svi.get_params(svi_state)['encoder$params'], f)
with open('params_chairs_encoder_beta5.pickle', 'wb') as f:
    pickle.dump(svi.get_params(svi_state)['decoder$params'], f)

# %%
#beta = 10
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
import numpy as npy
from google.colab.patches import cv2_imshow

from lib import load_dataset, VAEEncoder, VAEDecoder

# geometry Dataset
# with open('train_v2.npy', 'rb') as f:
#     data_x = np.load(f)



data_size, dim_feature = data_x.shape

image_size = 128
assert image_size * image_size == dim_feature

# Model

dim_z = 50
batch_size = 128
n_epochs = 200
learning_rate = 0.00005  # MNIST: 0.001
beta = 10

encoder_nn = VAEEncoder(image_size, dim_z)
decoder_nn = VAEDecoder(image_size)

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
    # plt.imsave(f'.results/epoch{epoch}.png', imgs, cmap='gray')
    npimgs = npy.array(imgs.tolist()) 
    cv2_imshow(npimgs)

for epoch in range(1, n_epochs + 1):
    time_start = time.time()

    key, subkey = rand.split(key)
    svi_state, total_loss = train_step(subkey, svi_state)

    key, subkey = rand.split(key)
    reconstruct_img(svi.get_params(svi_state), subkey, epoch)

    time_elapsed = time.time() - time_start
    print(f'Epoch {epoch}, loss {total_loss:.2f}, time {time_elapsed:.2f}s')

with open('params_chairs_encoder_beta10.pickle', 'wb') as f:
    pickle.dump(svi.get_params(svi_state)['encoder$params'], f)
with open('params_chairs_decoder_beta10.pickle', 'wb') as f:
    pickle.dump(svi.get_params(svi_state)['decoder$params'], f)

# %%
#beta = 15
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
import numpy as npy
from google.colab.patches import cv2_imshow

from lib import load_dataset, VAEEncoder, VAEDecoder

# geometry Dataset
# with open('train_v2.npy', 'rb') as f:
#     data_x = np.load(f)



data_size, dim_feature = data_x.shape

image_size = 128
assert image_size * image_size == dim_feature

# Model

dim_z = 50
batch_size = 128
n_epochs = 200
learning_rate = 0.00005  # MNIST: 0.001
beta = 15

encoder_nn = VAEEncoder(image_size, dim_z)
decoder_nn = VAEDecoder(image_size)

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
    # plt.imsave(f'.results/epoch{epoch}.png', imgs, cmap='gray')
    npimgs = npy.array(imgs.tolist()) 
    cv2_imshow(npimgs)

for epoch in range(1, n_epochs + 1):
    time_start = time.time()

    key, subkey = rand.split(key)
    svi_state, total_loss = train_step(subkey, svi_state)

    key, subkey = rand.split(key)
    reconstruct_img(svi.get_params(svi_state), subkey, epoch)

    time_elapsed = time.time() - time_start
    print(f'Epoch {epoch}, loss {total_loss:.2f}, time {time_elapsed:.2f}s')

with open('params_chairs_encoder_beta15.pickle', 'wb') as f:
    pickle.dump(svi.get_params(svi_state)['encoder$params'], f)
with open('params_chairs_decoder_beta15.pickle', 'wb') as f:
    pickle.dump(svi.get_params(svi_state)['decoder$params'], f)

# %%
#beta = 20
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
import numpy as npy
from google.colab.patches import cv2_imshow

from lib import load_dataset, VAEEncoder, VAEDecoder

# geometry Dataset
# with open('train_v2.npy', 'rb') as f:
#     data_x = np.load(f)



data_size, dim_feature = data_x.shape

image_size = 128
assert image_size * image_size == dim_feature

# Model

dim_z = 50
batch_size = 128
n_epochs = 200
learning_rate = 0.00005  # MNIST: 0.001
beta = 20

encoder_nn = VAEEncoder(image_size, dim_z)
decoder_nn = VAEDecoder(image_size)

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
    # plt.imsave(f'.results/epoch{epoch}.png', imgs, cmap='gray')
    npimgs = npy.array(imgs.tolist()) 
    cv2_imshow(npimgs)

for epoch in range(1, n_epochs + 1):
    time_start = time.time()

    key, subkey = rand.split(key)
    svi_state, total_loss = train_step(subkey, svi_state)

    key, subkey = rand.split(key)
    reconstruct_img(svi.get_params(svi_state), subkey, epoch)

    time_elapsed = time.time() - time_start
    print(f'Epoch {epoch}, loss {total_loss:.2f}, time {time_elapsed:.2f}s')

with open('params_chairs_encoder_beta20.pickle', 'wb') as f:
    pickle.dump(svi.get_params(svi_state)['encoder$params'], f)
with open('params_chairs_decoder_beta20.pickle', 'wb') as f:
    pickle.dump(svi.get_params(svi_state)['decoder$params'], f)

# %%
#beta = 1
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
import numpy as npy
from google.colab.patches import cv2_imshow

from lib import load_dataset, VAEEncoder, VAEDecoder

# geometry Dataset
# with open('train_v2.npy', 'rb') as f:
#     data_x = np.load(f)



data_size, dim_feature = data_x.shape

image_size = 128
assert image_size * image_size == dim_feature

# Model

dim_z = 50
batch_size = 128
n_epochs = 200
learning_rate = 0.00005  # MNIST: 0.001
beta = 1

encoder_nn = VAEEncoder(image_size, dim_z)
decoder_nn = VAEDecoder(image_size)

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
    # plt.imsave(f'.results/epoch{epoch}.png', imgs, cmap='gray')
    npimgs = npy.array(imgs.tolist()) 
    cv2_imshow(npimgs)

for epoch in range(1, n_epochs + 1):
    time_start = time.time()

    key, subkey = rand.split(key)
    svi_state, total_loss = train_step(subkey, svi_state)

    key, subkey = rand.split(key)
    reconstruct_img(svi.get_params(svi_state), subkey, epoch)

    time_elapsed = time.time() - time_start
    print(f'Epoch {epoch}, loss {total_loss:.2f}, time {time_elapsed:.2f}s')

with open('params_chairs_encoder_beta1.pickle', 'wb') as f:
    pickle.dump(svi.get_params(svi_state)['encoder$params'], f)
with open('params_chairs_decoder_beta1.pickle', 'wb') as f:
    pickle.dump(svi.get_params(svi_state)['decoder$params'], f)

# %%
#beta = 2
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
import numpy as npy
from google.colab.patches import cv2_imshow

from lib import load_dataset, VAEEncoder, VAEDecoder

# geometry Dataset
# with open('train_v2.npy', 'rb') as f:
#     data_x = np.load(f)



data_size, dim_feature = data_x.shape

image_size = 128
assert image_size * image_size == dim_feature

# Model

dim_z = 50
batch_size = 128
n_epochs = 200
learning_rate = 0.00005  # MNIST: 0.001
beta = 2

encoder_nn = VAEEncoder(image_size, dim_z)
decoder_nn = VAEDecoder(image_size)

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
    # plt.imsave(f'.results/epoch{epoch}.png', imgs, cmap='gray')
    npimgs = npy.array(imgs.tolist()) 
    cv2_imshow(npimgs)

for epoch in range(1, n_epochs + 1):
    time_start = time.time()

    key, subkey = rand.split(key)
    svi_state, total_loss = train_step(subkey, svi_state)

    key, subkey = rand.split(key)
    reconstruct_img(svi.get_params(svi_state), subkey, epoch)

    time_elapsed = time.time() - time_start
    print(f'Epoch {epoch}, loss {total_loss:.2f}, time {time_elapsed:.2f}s')

with open('params_chairs_encoder_beta2.pickle', 'wb') as f:
    pickle.dump(svi.get_params(svi_state)['encoder$params'], f)
with open('params_chairs_decoder_beta2.pickle', 'wb') as f:
    pickle.dump(svi.get_params(svi_state)['decoder$params'], f)

# %%
#beta = 1, z = 10
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
import numpy as npy
from google.colab.patches import cv2_imshow

from lib import load_dataset, VAEEncoder, VAEDecoder

# geometry Dataset
# with open('train_v2.npy', 'rb') as f:
#     data_x = np.load(f)



data_size, dim_feature = data_x.shape

image_size = 128
assert image_size * image_size == dim_feature

# Model

dim_z = 10
batch_size = 128
n_epochs = 200
learning_rate = 0.00005  # MNIST: 0.001
beta = 1

encoder_nn = VAEEncoder(image_size, dim_z)
decoder_nn = VAEDecoder(image_size)

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
    # plt.imsave(f'.results/epoch{epoch}.png', imgs, cmap='gray')
    npimgs = npy.array(imgs.tolist()) 
    cv2_imshow(npimgs)

for epoch in range(1, n_epochs + 1):
    time_start = time.time()

    key, subkey = rand.split(key)
    svi_state, total_loss = train_step(subkey, svi_state)

    key, subkey = rand.split(key)
    reconstruct_img(svi.get_params(svi_state), subkey, epoch)

    time_elapsed = time.time() - time_start
    print(f'Epoch {epoch}, loss {total_loss:.2f}, time {time_elapsed:.2f}s')

with open('params_chairs_encoder_beta1_z10.pickle', 'wb') as f:
    pickle.dump(svi.get_params(svi_state)['encoder$params'], f)
with open('params_chairs_decoder_beta1_z10.pickle', 'wb') as f:
    pickle.dump(svi.get_params(svi_state)['decoder$params'], f)

# %%
#beta = 2, z = 10
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
import numpy as npy
from google.colab.patches import cv2_imshow

from lib import load_dataset, VAEEncoder, VAEDecoder

# geometry Dataset
# with open('train_v2.npy', 'rb') as f:
#     data_x = np.load(f)



data_size, dim_feature = data_x.shape

image_size = 128
assert image_size * image_size == dim_feature

# Model

dim_z = 10
batch_size = 128
n_epochs = 200
learning_rate = 0.00005  # MNIST: 0.001
beta = 2

encoder_nn = VAEEncoder(image_size, dim_z)
decoder_nn = VAEDecoder(image_size)

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
    # plt.imsave(f'.results/epoch{epoch}.png', imgs, cmap='gray')
    npimgs = npy.array(imgs.tolist()) 
    cv2_imshow(npimgs)

for epoch in range(1, n_epochs + 1):
    time_start = time.time()

    key, subkey = rand.split(key)
    svi_state, total_loss = train_step(subkey, svi_state)

    key, subkey = rand.split(key)
    reconstruct_img(svi.get_params(svi_state), subkey, epoch)

    time_elapsed = time.time() - time_start
    print(f'Epoch {epoch}, loss {total_loss:.2f}, time {time_elapsed:.2f}s')

with open('params_chairs_encoder_beta2_z10.pickle', 'wb') as f:
    pickle.dump(svi.get_params(svi_state)['encoder$params'], f)
with open('params_chairs_decoder_beta2_z10.pickle', 'wb') as f:
    pickle.dump(svi.get_params(svi_state)['decoder$params'], f)

# %%
#beta = 5, z = 10
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
import numpy as npy
from google.colab.patches import cv2_imshow

from lib import load_dataset, VAEEncoder, VAEDecoder

# geometry Dataset
# with open('train_v2.npy', 'rb') as f:
#     data_x = np.load(f)



data_size, dim_feature = data_x.shape

image_size = 128
assert image_size * image_size == dim_feature

# Model

dim_z = 10
batch_size = 128
n_epochs = 200
learning_rate = 0.00005  # MNIST: 0.001
beta = 5

encoder_nn = VAEEncoder(image_size, dim_z)
decoder_nn = VAEDecoder(image_size)

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
    # plt.imsave(f'.results/epoch{epoch}.png', imgs, cmap='gray')
    npimgs = npy.array(imgs.tolist()) 
    cv2_imshow(npimgs)

for epoch in range(1, n_epochs + 1):
    time_start = time.time()

    key, subkey = rand.split(key)
    svi_state, total_loss = train_step(subkey, svi_state)

    key, subkey = rand.split(key)
    reconstruct_img(svi.get_params(svi_state), subkey, epoch)

    time_elapsed = time.time() - time_start
    print(f'Epoch {epoch}, loss {total_loss:.2f}, time {time_elapsed:.2f}s')

with open('params_chairs_encoder_beta5_z10.pickle', 'wb') as f:
    pickle.dump(svi.get_params(svi_state)['encoder$params'], f)
with open('params_chairs_decoder_beta5_z10.pickle', 'wb') as f:
    pickle.dump(svi.get_params(svi_state)['decoder$params'], f)

# %%
#beta = 10, z = 10
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
import numpy as npy
from google.colab.patches import cv2_imshow

from lib import load_dataset, VAEEncoder, VAEDecoder

# geometry Dataset
# with open('train_v2.npy', 'rb') as f:
#     data_x = np.load(f)



data_size, dim_feature = data_x.shape

image_size = 128
assert image_size * image_size == dim_feature

# Model

dim_z = 10
batch_size = 128
n_epochs = 200
learning_rate = 0.00005  # MNIST: 0.001
beta = 10

encoder_nn = VAEEncoder(image_size, dim_z)
decoder_nn = VAEDecoder(image_size)

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
    # plt.imsave(f'.results/epoch{epoch}.png', imgs, cmap='gray')
    npimgs = npy.array(imgs.tolist()) 
    cv2_imshow(npimgs)

for epoch in range(1, n_epochs + 1):
    time_start = time.time()

    key, subkey = rand.split(key)
    svi_state, total_loss = train_step(subkey, svi_state)

    key, subkey = rand.split(key)
    reconstruct_img(svi.get_params(svi_state), subkey, epoch)

    time_elapsed = time.time() - time_start
    print(f'Epoch {epoch}, loss {total_loss:.2f}, time {time_elapsed:.2f}s')

with open('params_chairs_encoder_beta10_z10.pickle', 'wb') as f:
    pickle.dump(svi.get_params(svi_state)['encoder$params'], f)
with open('params_chairs_decoder_beta10_z10.pickle', 'wb') as f:
    pickle.dump(svi.get_params(svi_state)['decoder$params'], f)

# %%
#beta = 20, z = 10
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
import numpy as npy
from google.colab.patches import cv2_imshow

from lib import load_dataset, VAEEncoder, VAEDecoder

# geometry Dataset
# with open('train_v2.npy', 'rb') as f:
#     data_x = np.load(f)



data_size, dim_feature = data_x.shape

image_size = 128
assert image_size * image_size == dim_feature

# Model

dim_z = 10
batch_size = 128
n_epochs = 200
learning_rate = 0.00005  # MNIST: 0.001
beta = 20

encoder_nn = VAEEncoder(image_size, dim_z)
decoder_nn = VAEDecoder(image_size)

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
    # plt.imsave(f'.results/epoch{epoch}.png', imgs, cmap='gray')
    npimgs = npy.array(imgs.tolist()) 
    cv2_imshow(npimgs)

for epoch in range(1, n_epochs + 1):
    time_start = time.time()

    key, subkey = rand.split(key)
    svi_state, total_loss = train_step(subkey, svi_state)

    key, subkey = rand.split(key)
    reconstruct_img(svi.get_params(svi_state), subkey, epoch)

    time_elapsed = time.time() - time_start
    print(f'Epoch {epoch}, loss {total_loss:.2f}, time {time_elapsed:.2f}s')

with open('params_chairs_encoder_beta20_z10.pickle', 'wb') as f:
    pickle.dump(svi.get_params(svi_state)['encoder$params'], f)
with open('params_chairs_decoder_beta20_z10.pickle', 'wb') as f:
    pickle.dump(svi.get_params(svi_state)['decoder$params'], f)

# %%

with open('svi_state.pickle', 'wb') as f:
    pickle.dump(svi_state, f)

# %%
#load model
import pickle

f = open('svi_state.pickle','rb')
svi_state = pickle.load(f)
f.close()

# %%
import jax
import jax.numpy as np
import jax.random as rand
import matplotlib.pyplot as plt
import numpyro.distributions as dist
import pickle

from lib import load_dataset,VAEEncoder, VAEDecoder
# from lib.model.py import 

jax.config.update('jax_platform_name', 'cpu')

# data_x = load_dataset(dataset='chairs')

data_size, dim_feature = data_x.shape

# image_size = 64
assert dim_feature == image_size * image_size

# dim_z = 100

key = rand.PRNGKey(42)
# params_encoder = svi.get_params(svi_state)['encoder$params']
# params_decoder = svi.get_params(svi_state)['decoder$params']

# encoder_nn = VAEEncoder(image_size, dim_z)
# decoder_nn = VAEDecoder(image_size)

# with open('params_chairs_encoder.pickle', 'rb') as f:
#     params_encoder = pickle.load(f)

# with open('params_chairs_decoder.pickle', 'rb') as f:
#     params_decoder = pickle.load(f)

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
idx = 500
x = data_x[idx][None, ...]

key, subkey = rand.split(key)
imgs = reconstruct(x, delta=2., target_dims=(100, 101))
# plt.imsave('1.png', imgs, cmap='gray')

import numpy as npy
npimgs = npy.array(imgs.tolist()) 
from google.colab.patches import cv2_imshow
cv2_imshow(npimgs)

# %%
npimgs.shape

# %%
plt.imsave(f'.results/traversal.png', imgs, cmap='gray')


# %%
# disentanglement metric
#load trained encoder
params_encoder = svi.get_params(svi_state)['encoder$params']
z_mean, z_var = encoder_nn.apply({'params': params_encoder}, data_x[0][None, ...])
print(z_mean.shape, z_var.shape)

# %%
# preprocess geometry Dataset for linear classification
import jax.numpy as np
from jax.numpy import load
# load dict of arrays
savedir = '/content/binzarized_position_x.npz'

dict_data = load(savedir)
# extract the first array
data_x = dict_data['arr_0']
data_x = data_x.reshape(data_x.shape[0], data_x.shape[1]*data_x.shape[2])

# %%
# preprocess geometry Dataset for linear classification
import jax.numpy as np
from jax.numpy import load
# load dict of arrays
savedir = '/content/binzarized_position_x.npz'

dict_data = load(savedir)
# extract the first array
data_x2 = dict_data['arr_0']
data_x2 = data_x2.reshape(data_x2.shape[0], data_x2.shape[1]*data_x2.shape[2])

# %%
data_x = np.concatenate((data_x, data_x2), axis=0)


# %%
data_x.shape

# %%
z_diff = []
bs = 200
for i in range(int(data_x.shape[0]/bs)):
  print(i)
  z_means, z_vars = encoder_nn.apply({'params': params_encoder}, data_x[i*bs:(i+1)*bs])
  z_diff.append(z_means)

z_diff_x = z_diff[0]
for i in range(1, len(z_diff)):
  z_diff_x = np.concatenate((z_diff_x, z_diff[i]), axis=0)

z_diff_x = npy.asarray(z_diff_x)
z_diff_x = torch.FloatTensor(z_diff_x)
# torch_ten = torch.from_numpy(z_diff_x)
z_diff_x.shape

# %%
z_diff_y1 = npy.zeros((1000)).astype(int)
z_diff_y2 = (npy.zeros((1000)) +1).astype(int)
z_diff_y = npy.concatenate((z_diff_y1, z_diff_y2), axis=0)
z_diff_y = list(z_diff_y)
print(z_diff_y)
z_diff_y = torch.Tensor(z_diff_y)
print(z_diff_y)
# z_diff_y = z_diff_y.int()
z_diff_y.shape 

# %%
print(devices)

# %%
#load model encoder
import pickle

from lib import load_dataset, VAEEncoder, VAEDecoder

# f = open('params_chairs_encoder.pickle','rb')

# svi.get_params(svi_state)['encoder$params'] = pickle.load(f)

# f.close()
# params_encoder = svi.get_params(svi_state)['encoder$params']
# f = open('params_chairs_decoder.pickle','rb')
# svi.get_params(svi_state)['decoder$params'] = pickle.load(f)
# f.close()


with open('params_chairs_encoder_beta10_z10.pickle', 'rb') as f:
    params_encoder = pickle.load(f)

image_size = 128
dim_z = 10
encoder_nn = VAEEncoder(image_size, dim_z)
decoder_nn = VAEDecoder(image_size)

# %%
# preprocess geometry Dataset for linear classification
import jax.numpy as np
from jax.numpy import load
# load dict of arrays
savedir = '/content/binzarized_scale.npz'

dict_data = load(savedir)
# extract the first array
data_x = dict_data['arr_0']
data_x = data_x.reshape(data_x.shape[0], data_x.shape[1]*data_x.shape[2])

# %%
data_x.shape[0]

# %%
z_diff = []
bs = 200
for i in range(int(data_x.shape[0]/bs)):
  print(i)
  params_encoder = svi.get_params(svi_state)['encoder$params']
  z_means, z_vars = encoder_nn.apply({'params': params_encoder}, data_x[i*bs:(i+1)*bs])

  z_means_even = z_means[0::2]
  z_means_odd = z_means[1::2]
  temp = abs(z_means_odd - z_means_even)
  z_diff.append(temp)



# %%
z_diff[0].shape

# %%
#for 1st feature data
z_diff_x1 = z_diff[0]
for i in range(1, len(z_diff)):
  z_diff_x1 = np.concatenate((z_diff_x1, z_diff[i]), axis=0)
z_diff_x1.shape

# %%
#for 2nd feature data

z_diff_x2 = z_diff[0]
for i in range(1, len(z_diff)):
  z_diff_x2 = np.concatenate((z_diff_x2, z_diff[i]), axis=0)
z_diff_x2.shape

# %%
#for 3nd feature data

z_diff_x3 = z_diff[0]
for i in range(1, len(z_diff)):
  z_diff_x3 = np.concatenate((z_diff_x3, z_diff[i]), axis=0)
z_diff_x3.shape

# %%
#for 4th feature data

z_diff_x4 = z_diff[0]
for i in range(1, len(z_diff)):
  z_diff_x4 = np.concatenate((z_diff_x4, z_diff[i]), axis=0)
z_diff_x4.shape

# %%
import numpy as npy
import torch
z_diff_x = np.concatenate((z_diff_x1, z_diff_x2), axis=0)
# z_diff_x = np.concatenate((z_diff_x1, z_diff_x2,z_diff_x3,z_diff_x4), axis=0)

z_diff_x = npy.asarray(z_diff_x)
z_diff_x = torch.FloatTensor(z_diff_x)
# torch_ten = torch.from_numpy(z_diff_x)
z_diff_x.shape

# %%
z_diff_x[2]

# %%
z_diff_y1 = (npy.zeros((3000)) +0).astype(int)
z_diff_y2 = (npy.zeros((3000)) +1).astype(int)
# z_diff_y3 = (npy.zeros((3000)) +0).astype(int)
# z_diff_y4 = (npy.zeros((3000)) +0).astype(int)

z_diff_y = npy.concatenate((z_diff_y1, z_diff_y2), axis=0)
# z_diff_y = npy.concatenate((z_diff_y1, z_diff_y2,z_diff_y3,z_diff_y4), axis=0)

z_diff_y = list(z_diff_y)
# z_diff_y = npy.array(z_diff_y)
print(z_diff_y)
z_diff_y = torch.tensor(z_diff_y)
print(z_diff_y)
# z_diff_y = z_diff_y.int()
z_diff_y.shape 

# %%
import torch 
import matplotlib.pyplot as plt
import numpy as np

import torch
import torchvision
import torchvision.transforms as transforms

# %%
class LinearClassifier(torch.nn.Module):
  def __init__(self, input_dim=50, output_dim=2):
    super(LinearClassifier, self).__init__()
    self.linear = torch.nn.Linear(input_dim, output_dim)
  
  def forward(self, x):
    return self.linear(x)

# %%
model = LinearClassifier()
# criterion = torch.nn.CrossEntropyLoss()

criterion = torch.nn.CrossEntropyLoss()

optimizer = torch.optim.SGD(model.parameters(), lr=0.1)

# %%
testt = npy.zeros((2000,50))
for i in range(1000):
  testt[i,:] = i
for i in range(1000,2000):
  testt[i,:] = i+500
testt = torch.FloatTensor(testt)
testt[995:1005]

# %%
all_loss = []
acc=[]
for epoch in range(5000):
  output = model(z_diff_x)

  loss = criterion(output, z_diff_y)
  all_loss.append(loss.item())
  loss.backward()

  optimizer.step()
  optimizer.zero_grad()
  if epoch%100 == 0:
    print(epoch)
    prediction = output.argmax(dim=1, keepdim=True)
    prediction.eq(z_diff_y.view_as(prediction)).sum().item()
    acc.append(prediction.eq(z_diff_y.view_as(prediction)).sum().item()/z_diff_y.shape[0])


# %%
plt.plot(acc)

# %%
acc[-10:]

# %%
( 0.5945 + 0.6233333333333333  + 0.6806666666666666  + 0.6492  + 0.6151666666666666 + 0.6885 )/6

# %%
#beta = 1 (all 4 features, z=10) 
plt.plot(all_loss)

# %%
#beta = 1 (all 4 features, z=10) 

acc[-10:]

# %%
#beta = 1 (all 4 features, z=10)
plt.plot(acc)

# %%
#beta = 1 (all 4 features, z=10)

acc[-10:]

# %%
#beta = 1 (all 4 features, z=50) 
plt.plot(all_loss)

# %%
#beta = 1 (all 4 features, z=50)
plt.plot(acc)

# %%
#beta = 5 (all 4 features, z=50)
plt.plot(all_loss)

# %%
#beta = 5 (all 4 features, z=50)
plt.plot(acc)

# %%
#beta = 10 (all 4 features, z=50)
plt.plot(all_loss)

# %%
#beta = 10 (all 4 features, z=50)

all_loss[-10:]

# %%
#beta = 10 (all 4 features, z =50)
plt.plot(acc)

# %%
#beta = 10 (all 4 features, z=50)

acc[-10:]

# %%
#beta = 20 (all 4 features, z=50)
plt.plot(all_loss)

# %%
#beta = 20 (all 4 features, z=50)

all_loss[-10:]

# %%
#beta = 20 (all 4 features, z =50)
plt.plot(acc)

# %%
#beta = 20 (all 4 features, z=50)

acc[-10:]

# %%
#beta = 1 (position x and position y)
plt.plot(all_loss)

# %%
#beta = 1 (position x and position y)
plt.plot(acc)

# %%
#beta = 10 (position x and position y)
plt.plot(all_loss)

# %%
#beta = 10 (position x and position y)
plt.plot(acc)

# %%
all_loss[-10:]

# %%



