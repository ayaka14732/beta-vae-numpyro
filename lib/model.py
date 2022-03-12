import flax.linen as nn
import jax.numpy as np

class VAEEncoder(nn.Module):
    # image_size: int
    dim_z: int

    @nn.compact
    def __call__(self, x: np.ndarray):
        batch_size, _, _, _ = x.shape

        x = nn.softplus(nn.Conv(16, (4, 4), 1)(x))  # (b, 216, 176, 16)
        x = nn.softplus(nn.Conv(32, (4, 4), 1)(x))  # (b, 216, 176, 32)
        x = nn.softplus(nn.Conv(32, (4, 4), 2)(x))  # (b, 108, 88, 32)
        x = nn.softplus(nn.Conv(64, (4, 4), 2)(x))  # (b, 54, 44, 64)
        x = x.reshape(batch_size, -1)  # (b, 54 * 44 * 64)

        z_loc = nn.Dense(self.dim_z)(x)
        z_std = np.exp(nn.Dense(self.dim_z)(x))

        return z_loc, z_std

class VAEDecoder(nn.Module):
    # image_size: int

    @nn.compact
    def __call__(self, z: np.ndarray):
        batch_size, _ = z.shape
        z = nn.softplus(nn.Dense(54 * 44 * 64)(z))  # (b, 54 * 44 * 64)
        z = z.reshape(batch_size, 54, 44, 64)  # (b, 54, 44, 64)
        z = nn.softplus(nn.ConvTranspose(32, (4, 4), (2, 2))(z))  # (b, 108, 88, 32)
        z = nn.softplus(nn.ConvTranspose(32, (4, 4), (2, 2))(z))  # (b, 216, 176, 32)
        z = nn.softplus(nn.ConvTranspose(16, (4, 4))(z))  # (b, 216, 176, 16)
        z = nn.sigmoid(nn.ConvTranspose(3, (4, 4))(z))  # (b, 216, 176, 3)
        return z
