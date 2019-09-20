import tensorflow as tf
import copy
import numpy as np

def gaussian_likelihood(x, mu, log_std):
    pre_sum = -0.5*(((x-mu)/(tf.exp(log_std)+1e-8))**2 + 2*log_std + np.log(2*np.pi))
    return tf.reduce_sum(pre_sum, axis=1)

def logit(x):
    return - tf.log(1. / x - 1.)

def get_gaes(rewards, dones, values, next_values, gamma, _lambda, normalize):
    deltas = [r + gamma * (1 - d) * nv - v for r, d, nv, v in zip(rewards, dones, next_values, values)]
    deltas = np.stack(deltas)
    gaes = copy.deepcopy(deltas)
    for t in reversed(range(len(deltas) - 1)):
        gaes[t] = gaes[t] + (1 - dones[t]) * gamma * _lambda * gaes[t + 1]
    target = gaes + values

    if normalize:
        gaes = (gaes - gaes.mean()) / (gaes.std() + 1e-8)
    return gaes, target