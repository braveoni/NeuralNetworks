import numpy as np


def sigmoid(x):
    return 1 / (1 + np.exp(-x))


def derivative_sigmoid(x):
    return (1 - x) * x


def softmax(x):
    return np.exp(x) / sum(np.exp(x))


def relu(x):
    return np.where(x > 0, x, 0)


def tangent(x):
    return (np.exp(x) - np.exp(-x))/(np.exp(x) + np.exp(-x))


def derivative_tangent(x):
    return 1 - np.square(x)


def mse(p, y):
    return np.average(np.square(p - y))


def mae(p, y):
    return np.average(np.abs(p - y))


def cat_cross_entropy(p, y):
    return -sum((y * np.log2(p) + (1 - y) * np.log2(1 - p)))
