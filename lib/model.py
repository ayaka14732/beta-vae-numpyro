import flax.linen as nn
import jax.numpy as np

class VAEEncoder(nn.Module):
    image_size: int
    dim_z: int

    @nn.compact
    def __call__(self, x: np.ndarray):
        batch_size, _ = x.shape
        x = x.reshape(batch_size, self.image_size, self.image_size, 1)  # (b, 128, 128, 1)
        x = nn.softplus(nn.Conv(16, (4, 4), 1)(x))  # (b, 128, 128, 16)
        x = nn.softplus(nn.Conv(32, (4, 4), 2)(x))  # (b, 64, 64, 32)
        x = nn.softplus(nn.Conv(64, (4, 4), 2)(x))  # (b, 32, 32, 64)
        x = x.reshape(batch_size, -1)  # (b, 32 * 32 * 64)

        z_loc = nn.Dense(self.dim_z)(x)
        z_std = np.exp(nn.Dense(self.dim_z)(x))

        return z_loc, z_std

class VAEDecoder(nn.Module):
    image_size: int

    @nn.compact
    def __call__(self, z: np.ndarray):
        batch_size, _ = z.shape
        z = nn.softplus(nn.Dense(32 * 32 * 64)(z))
        z = z.reshape(batch_size, 32, 32, 64)  # (b, 32, 32, 64)
        z = nn.softplus(nn.ConvTranspose(32, (4, 4), (2, 2))(z))  # (b, 64, 64, 32)
        z = nn.softplus(nn.ConvTranspose(16, (4, 4))(z))  # (b, 128, 128, 16)
        z = nn.softplus(nn.ConvTranspose(1, (4, 4))(z))  # (b, 128, 128, 1)
        z = z.reshape(batch_size, -1)  # (b, 128 * 128)
        z = nn.sigmoid(nn.Dense(self.image_size * self.image_size)(z))
        return z
