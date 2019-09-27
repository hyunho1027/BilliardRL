import tensorflow as tf
import copy
import numpy as np

def logit(x):
    return - tf.log(1. / x - 1.)

def clip_b4_exp(x):
    return tf.clip_by_value(x,-10,10)

def gaussian_likelihood(x, mu, log_std):
    pre_sum = -0.5*(((x-mu)/(tf.exp(log_std)+1e-8))**2 + 2*log_std + np.log(2*np.pi))
    return tf.reduce_sum(pre_sum, axis=1)


def get_gaes(rs, ds, vs, next_vs, gamma, _lambda, normalize):
    deltas = [r + gamma * (1 - d) * nv - v for r, d, nv, v in zip(rs, ds, next_vs, vs)]
    deltas = np.stack(deltas)
    gaes = copy.deepcopy(deltas)
    for t in reversed(range(len(deltas) - 1)):
        gaes[t] = gaes[t] + (1 - ds[t]) * gamma * _lambda * gaes[t + 1]
    target = gaes + vs

    if normalize:
        gaes = (gaes - gaes.mean()) / (gaes.std() + 1e-8)
    return gaes, target


def vec_data_augmentation(s, a, next_s):
    s, a, next_s = map(np.copy, [s, a, next_s])
    s_augs, a_augs, next_s_augs = [], [], []
    for i in range(4):
        if i>0:
            quad = 1
            while a[0]>=1/4 and quad<4:
                a[0] -= 1/4
                quad += 1
            a[0] *= -1
            if i%2 != quad%2 : 
                a[0] += 1/2
            a[0] += ((quad-1)/4)
            a[0] = a[0] -(a[0]>1) +(a[0]<0)
            for j in range(i%2,8,2):
                s[j] *= -1
        for j in range(2):
            if j>0:
                s[4:6], s[6:8], next_s[4:6], next_s[6:8] = s[6:8], s[4:6], next_s[6:8], next_s[4:6]
            
            s, a, next_s = map(np.copy, [s, a, next_s])
            s_augs.append(s)
            a_augs.append(a)
            next_s_augs.append(next_s)
    
    return s_augs, a_augs, next_s_augs

def vis_data_augmentation(s, a, next_s):
    s, a, next_s = map(np.copy, [s, a, next_s])
    s_augs, a_augs, next_s_augs = [], [], []
    for i in range(4):
        if i>0:
            quad = 1
            while a[0]>=1/4 and quad<4:
                a[0] -= 1/4
                quad += 1
            a[0] *= -1
            if i%2 != quad%2 : 
                a[0] += 1/2
            a[0] += ((quad-1)/4)
            a[0] = a[0] -(a[0]>1) +(a[0]<0)
            
            s = np.flipud(s) if i%2 else np.fliplr(s)
            next_s = np.flipud(next_s) if i%2 else np.fliplr(next_s)
        
        s, a, next_s = map(np.copy, [s, a, next_s])
        s_augs.append(s)
        a_augs.append(a)
        next_s_augs.append(next_s)

    return s_augs, a_augs, next_s_augs
