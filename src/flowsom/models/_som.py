from __future__ import annotations

import numpy as np
from numba import jit


@jit(nopython=True, parallel=False)
def eucl(p1, p2):
    distance = 0.0
    for j in range(len(p1)):
        diff = p1[j] - p2[j]
        distance += diff * diff
    return np.sqrt(distance)


@jit(nopython=True)
def manh(p1, p2):
    return np.sum(np.abs(p1 - p2))



@jit(nopython=True)
def SOM(data, codes, nhbrdist, alphas, radii, ncodes, rlen, distf=eucl, seed=None):
    if seed is not None:
        np.random.seed(seed)
    xdists = np.zeros(ncodes)
    n = data.shape[0]
    px = data.shape[1]
    niter = rlen * n
    threshold = radii[0]
    thresholdStep = (radii[0] - radii[1]) / niter
    change = 1.0

    # Match R's som.c convergence: when change < 1, R sets k = niter.
    # The loop body executes once more (with k=niter for alpha computation),
    # then k++ causes the loop condition to fail, exiting.
    exit_after_this = False
    for k in range(niter):
        if exit_after_this:
            break
        if k % n == 0:
            if change < 1:
                exit_after_this = True
            change = 0.0

        i = np.random.randint(n)

        nearest = 0
        for cd in range(ncodes):
            xdists[cd] = distf(data[i, :], codes[cd, :])
            if xdists[cd] < xdists[nearest]:
                nearest = cd

        if threshold < 1.0:
            threshold = 0.5
        # When converging, R uses k=niter for alpha (always yields alphas[1])
        effective_k = niter if exit_after_this else k
        alpha = alphas[0] - (alphas[0] - alphas[1]) * effective_k / niter

        for cd in range(ncodes):
            if nhbrdist[cd, nearest] > threshold:
                continue

            for j in range(px):
                tmp = data[i, j] - codes[cd, j]
                change += abs(tmp)
                codes[cd, j] += tmp * alpha

        threshold -= thresholdStep
    return codes


@jit(nopython=True)
def map_data_to_codes(data, codes, distf=eucl):
    n_codes = codes.shape[0]
    nd = data.shape[0]
    nn_codes = np.zeros(nd)
    nn_dists = np.zeros(nd)
    for i in range(nd):
        minid = -1
        mindist = np.inf
        for cd in range(n_codes):
            tmp = distf(data[i, :], codes[cd, :])
            if tmp < mindist:
                mindist = tmp
                minid = cd
        nn_codes[i] = minid
        nn_dists[i] = mindist
    return nn_codes, nn_dists
