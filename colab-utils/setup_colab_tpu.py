# https://github.com/ayaka14732/tpu-starter/blob/main/00-basics/setup_colab_tpu.py

# Note: This script is for Colab TPU only.
# If you are using Cloud TPU VMs, you don't need to run it.

import jax
from jax.tools.colab_tpu import setup_tpu

setup_tpu()

devices = jax.devices()
print(devices)
