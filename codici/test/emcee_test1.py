#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Apr 14 11:59:40 2023

@author: niccolopassaleva
"""


import numpy as np
import emcee
import zeus
import corner
import matplotlib.pyplot as plt
from IPython.display import display, Math

#%% Setting the problem
def log_prob(x, mu, cov):
    diff = x - mu
    return -0.5 * np.dot(diff, np.linalg.solve(cov, diff))

ndim = 5

np.random.seed(42)
means = np.random.rand(ndim)

cov = 0.5 - np.random.rand(ndim**2).reshape((ndim, ndim))
cov = np.triu(cov)
cov += cov.T - np.diag(cov.diagonal())
cov = np.dot(cov, cov)

nwalkers = 32
p0 = np.random.rand(nwalkers, ndim)

labels = ["x1", "x2", "x3", "x4", "x5"]

#%% emcee
esampler = emcee.EnsembleSampler(nwalkers, ndim, log_prob, args=[means, cov])

estate = esampler.run_mcmc(p0, 100)
esampler.reset()

esampler.run_mcmc(estate, 10000)

esamples = esampler.get_chain(flat=True)
fig1 = corner.corner(esamples, labels=labels)

for i in range(ndim):
    mcmc = np.percentile(esamples[:, i], [16, 50, 84])
    q = np.diff(mcmc)
    txt = "\mathrm{{{3}}} = {0:.3f}_{{-{1:.3f}}}^{{{2:.3f}}}"
    txt = txt.format(mcmc[1], q[0], q[1], labels[i])
    display(Math(txt))

#%% zeus
zsampler = zeus.EnsembleSampler(nwalkers, ndim, log_prob, args=[means, cov])

zstate = zsampler.run_mcmc(p0, 100)
zsampler.reset()

zsampler.run_mcmc(zstate, 10000)

zsamples = zsampler.get_chain(flat=True)
fig2 = corner.corner(zsamples, labels=labels)

for i in range(ndim):
    mcmc = np.percentile(zsamples[:, i], [16, 50, 84])
    q = np.diff(mcmc)
    txt = "\mathrm{{{3}}} = {0:.3f}_{{-{1:.3f}}}^{{{2:.3f}}}"
    txt = txt.format(mcmc[1], q[0], q[1], labels[i])
    display(Math(txt))

#plt.hist(samples[:, 2], 100, color="k", histtype="step")
#plt.xlabel(r"$\theta_1$")
#plt.ylabel(r"$p(\theta_1)$")
#plt.gca().set_yticks([]);
