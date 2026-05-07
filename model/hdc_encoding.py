import numpy as np

DIM = 1000

def encode_sample(sample):
    hv = np.zeros(DIM)
    for value in sample:
        rand_vec = np.random.choice([-1, 1], size=DIM)
        hv += value * rand_vec
    return np.sign(hv)