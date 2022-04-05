import flax.linen as nn
import jax.numpy as np

class VAEEncoder(nn.Module):
    # image_size: int
    dim_z: int

    @nn.compact
    def __call__(self, x: np.ndarray):
        batch_size, _, _, _ = x.shape

        x  # (b, 208, 176, 3)

        x = nn.Conv(32, (5, 5), 2)(x)  # (b, 104, 88, 32)
        # x = nn.BatchNorm(use_running_average=False)(x)
        x = nn.softplus(x)

        x = nn.softplus(nn.Conv(64, (5, 5), 2)(x))  # (b, 52, 44, 64)
        # x = nn.BatchNorm(use_running_average=False)(x)
        x = nn.softplus(x)

        x = nn.softplus(nn.Conv(128, (5, 5), 2)(x))  # (b, 26, 22, 128)
        # x = nn.BatchNorm(use_running_average=False)(x)
        x = nn.softplus(x)

        x = nn.softplus(nn.Conv(256, (5, 5), 2)(x))  # (b, 13, 11, 256)
        # x = nn.BatchNorm(use_running_average=False)(x)
        x = nn.softplus(x)

        x = x.reshape(batch_size, -1)  # (b, 13 * 11 * 256)

        z_loc = nn.Dense(self.dim_z)(x)
        z_std = np.exp(nn.Dense(self.dim_z)(x))

        return z_loc, z_std

class VAEDecoder(nn.Module):
    # image_size: int

    @nn.compact
    def __call__(self, z: np.ndarray):
        batch_size, _ = z.shape
        z = nn.softplus(nn.Dense(13 * 11 * 256)(z))  # (b, 13 * 11 * 256)
        z = z.reshape(batch_size, 13, 11, 256)  # (b, 13, 11, 256)

        z = nn.ConvTranspose(128, (5, 5), (2, 2))(z)  # (b, 26, 22, 128)
        # z = nn.BatchNorm(use_running_average=False)(z)
        z = nn.softplus(z)

        z = nn.ConvTranspose(64, (5, 5), (2, 2))(z)  # (b, 52, 44, 64)
        # z = nn.BatchNorm(use_running_average=False)(z)
        z = nn.softplus(z)

        z = nn.ConvTranspose(32, (5, 5), (2, 2))(z)  # (b, 104, 88, 32)
        # z = nn.BatchNorm(use_running_average=False)(z)
        z = nn.softplus(z)

        z = nn.ConvTranspose(3, (5, 5), (2, 2))(z)  # (b, 208, 176, 3)
        # z = nn.BatchNorm(use_running_average=False)(z)
        z = nn.sigmoid(z)

        return z
