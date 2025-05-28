
import numpy as np

def build_observation(obs_buffer):
    return np.concatenate(list(obs_buffer)).astype(np.float32)
