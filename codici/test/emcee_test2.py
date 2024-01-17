#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Apr 14 15:33:57 2023

@author: niccolopassaleva
"""


import numpy as np
import emcee
import zeus
import dynesty
from dynesty import plotting as dyplot
import matplotlib.pyplot as plt
from scipy.optimize import minimize
import corner
from IPython.display import display, Math
from chainconsumer import ChainConsumer

#%% Setting the problem
np.random.seed(123)

# Choose the "true" parameters.
m_true = -0.9594
b_true = 4.294
f_true = 0.534

# Generate some synthetic data from the model.
N = 50
x = np.sort(10 * np.random.rand(N))
yerr = 0.1 + 0.5 * np.random.rand(N)
y = m_true * x + b_true
y += np.abs(f_true * y) * np.random.randn(N)
y += yerr * np.random.randn(N)

plt.errorbar(x, y, yerr=yerr, fmt=".k", capsize=0)
x0 = np.linspace(0, 10, 500)
plt.plot(x0, m_true * x0 + b_true, "k", alpha=0.3, lw=3)
plt.xlim(0, 10)
plt.xlabel("x")
plt.ylabel("y")

#%% Least squares
A = np.vander(x, 2)
C = np.diag(yerr * yerr)
ATA = np.dot(A.T, A / (yerr**2)[:, None])
cov = np.linalg.inv(ATA)
w = np.linalg.solve(ATA, np.dot(A.T, y / yerr**2))

print("Least-squares estimates:")
print("m = {0:.3f} ± {1:.3f}".format(w[0], np.sqrt(cov[0, 0])))
print("b = {0:.3f} ± {1:.3f}".format(w[1], np.sqrt(cov[1, 1])))

plt.errorbar(x, y, yerr=yerr, fmt=".k", capsize=0)
plt.plot(x0, m_true * x0 + b_true, "k", alpha=0.3, lw=3, label="truth")
plt.plot(x0, np.dot(np.vander(x0, 2), w), "--k", label="LS")
plt.legend(fontsize=14)
plt.xlim(0, 10)
plt.xlabel("x")
plt.ylabel("y")

#%% Maximum likelihood
def log_likelihood(theta, x, y, yerr):
    m, b, log_f = theta
    model = m * x + b
    sigma2 = yerr**2 + model**2 * np.exp(2 * log_f)
    return -0.5 * np.sum((y - model) ** 2 / sigma2 + np.log(sigma2))

np.random.seed(42)
nll = lambda *args: -log_likelihood(*args)
initial = np.array([m_true, b_true, np.log(f_true)]) + 0.1 * np.random.randn(3)
soln = minimize(nll, initial, args=(x, y, yerr))
m_ml, b_ml, log_f_ml = soln.x

print("Maximum likelihood estimates:")
print("m = {0:.3f}".format(m_ml))
print("b = {0:.3f}".format(b_ml))
print("f = {0:.3f}".format(np.exp(log_f_ml)))

fig1 = plt.errorbar(x, y, yerr=yerr, fmt=".k", capsize=0)
plt.plot(x0, m_true * x0 + b_true, "k", alpha=0.3, lw=3, label="truth")
plt.plot(x0, np.dot(np.vander(x0, 2), w), "--k", label="LS")
plt.plot(x0, np.dot(np.vander(x0, 2), [m_ml, b_ml]), ":k", label="ML")
plt.legend(fontsize=14)
plt.xlim(0, 10)
plt.xlabel("x")
plt.ylabel("y")

#%% preparation for MCMC
def log_prior(theta):
    lp = 0.
    m, b, log_f = theta
    
    if -0.0 < b < 10.0 and -10.0 < log_f < 1.0:
        lp = 0.
    else:
        lp = -np.inf
        
    mm = -1.
    msigma = 0.2
    lp -= 0.5*((m - mm)/msigma)**2
    
    return lp

def log_probability(theta, x, y, yerr):
    lp = log_prior(theta)
    if not np.isfinite(lp):
        return -np.inf
    return lp + log_likelihood(theta, x, y, yerr)

pos = soln.x + 1e-4 * np.random.randn(32, 3)
nwalkers, ndim = pos.shape

#%% MCMC with emcee
sampler = emcee.EnsembleSampler(nwalkers, ndim, log_probability, args=(x, y, yerr))
sampler.run_mcmc(pos, 5000, progress=True)

fig2, axes = plt.subplots(3, figsize=(10, 7), sharex=True)
samples = sampler.get_chain()
labels = ["m", "b", "log(f)"]
for i in range(ndim):
    ax = axes[i]
    ax.plot(samples[:, :, i], "k", alpha=0.3)
    ax.set_xlim(0, len(samples))
    ax.set_ylabel(labels[i])
    ax.yaxis.set_label_coords(-0.1, 0.5)

axes[-1].set_xlabel("step number")

tau = sampler.get_autocorr_time()
print(tau)

flat_samples = sampler.get_chain(discard=100, thin=15, flat=True)
print(flat_samples.shape)

#plotting results
fig3 = corner.corner(flat_samples, labels=labels, truths=[m_true, b_true, np.log(f_true)])

inds = np.random.randint(len(flat_samples), size=100)
plt.figure(4)
for ind in inds:
    sample = flat_samples[ind]
    plt.plot(x0, np.dot(np.vander(x0, 2), sample[:2]), "C1", alpha=0.1)
    
plt.errorbar(x, y, yerr=yerr, fmt=".k", capsize=0)
plt.plot(x0, m_true * x0 + b_true, "k", label="truth")
plt.legend(fontsize=14)
plt.xlim(0, 10)
plt.xlabel("x")
plt.ylabel("y");

for i in range(ndim):
    mcmc = np.percentile(flat_samples[:, i], [16, 50, 84])
    q = np.diff(mcmc)
    txt = "\mathrm{{{3}}} = {0:.3f}_{{-{1:.3f}}}^{{{2:.3f}}}"
    txt = txt.format(mcmc[1], q[0], q[1], labels[i])
    display(Math(txt))

#%% MCMC with zeus
zsampler = zeus.EnsembleSampler(nwalkers, ndim, log_probability, args=(x, y, yerr))
zsampler.run_mcmc(pos, 5000, progress=True)

fig5, axes = plt.subplots(3, figsize=(10, 7), sharex=True)
zsamples = zsampler.get_chain()
labels = ["m", "b", "log(f)"]
for i in range(ndim):
    ax = axes[i]
    ax.plot(zsamples[:, :, i], "k", alpha=0.3)
    ax.set_xlim(0, len(zsamples))
    ax.set_ylabel(labels[i])
    ax.yaxis.set_label_coords(-0.1, 0.5)

axes[-1].set_xlabel("step number")

tau_z = zeus.AutoCorrTime(zsamples)
print(tau_z)

flat_zsamples = zsampler.get_chain(discard=100, thin=15, flat=True)
print(flat_zsamples.shape)

#plotting results
fig6 = corner.corner(flat_zsamples, labels=labels, truths=[m_true, b_true, np.log(f_true)])

inds = np.random.randint(len(flat_samples), size=100)
plt.figure(7)
for ind in inds:
    zsample = flat_zsamples[ind]
    plt.plot(x0, np.dot(np.vander(x0, 2), zsample[:2]), "C1", alpha=0.1)
    
plt.errorbar(x, y, yerr=yerr, fmt=".k", capsize=0)
plt.plot(x0, m_true * x0 + b_true, "k", label="truth")
plt.legend(fontsize=14)
plt.xlim(0, 10)
plt.xlabel("x")
plt.ylabel("y");

for i in range(ndim):
    mcmc = np.percentile(flat_zsamples[:, i], [16, 50, 84])
    q = np.diff(mcmc)
    txt = "\mathrm{{{3}}} = {0:.3f}_{{-{1:.3f}}}^{{{2:.3f}}}"
    txt = txt.format(mcmc[1], q[0], q[1], labels[i])
    display(Math(txt))
    
#%% Dynesty nested sampling

rstate= np.random.default_rng(56101)

def log_like(theta):
    m, b, log_f = theta
    model = m * x + b
    sigma2 = yerr**2 + model**2 * np.exp(2 * log_f)
    return -0.5 * np.sum((y - model) ** 2 / sigma2 + np.log(sigma2))

# prior transform
def prior_transform(utheta):
    um, ub, ulf = utheta
    m = 5.5 * um - 5.
    b = 10. * ub
    lnf = 11. * ulf - 10.
    
    return m, b, lnf

dsampler = dynesty.DynamicNestedSampler(log_like, prior_transform, ndim=3,
                                        bound='multi', sample='rwalk', rstate=rstate)
dsampler.run_nested()
dres = dsampler.results

truths = [m_true, b_true, np.log(f_true)]
labels = [r'$m$', r'$b$', r'$\ln f$']
fig, axes = dyplot.traceplot(dsampler.results, truths=truths, labels=labels,
                             fig=plt.subplots(3, 2, figsize=(16, 12)))
fig.tight_layout()



fig, axes = dyplot.cornerplot(dres, truths=truths, show_titles=True, 
                              title_kwargs={'y': 1.04}, labels=labels,
                              fig=plt.subplots(3, 3, figsize=(15, 15)))

#%%
for i in range(ndim):
    q=np.array([0.16,0.5,0.84])
    sampx = np.atleast_1d(dres.samples[:,i])
    weights=dres.importance_weights()
    weights = np.atleast_1d(weights)
    idx = np.argsort(sampx)  # sort samples
    sw = weights[idx]  # sort weights
    cdf = np.cumsum(sw)[:-1]  # compute CDF
    cdf /= cdf[-1]  # normalize CDF
    cdf = np.append(0, cdf)  # ensure proper span
    quantiles = np.interp(q, cdf, sampx[idx]).tolist()
    err = np.diff(quantiles)
    txt = "\mathrm{{{3}}} = {0:.3f}_{{-{1:.3f}}}^{{{2:.3f}}}"
    txt = txt.format(quantiles[1], err[0], err[1], labels[i])
    display(Math(txt))


#%%

c = ChainConsumer()
c.add_chain(dres.samples, parameters=labels, name='dynesty')
c.add_chain(flat_zsamples, parameters=labels, name='zeus')
c.add_chain(flat_samples, parameters=labels, name='emcee')

c.configure(legend_artists=True)

fig = c.plotter.plot_distributions(truth=truths)

fig.set_size_inches(3 + fig.get_size_inches())

#%%

sampx = np.atleast_1d(dres.samples[:,i])
a,=sampx.shape

dynesty_chain=np.zeros((a, ndim))

for i in range(ndim):
    sampx = np.atleast_1d(dres.samples[:,i])
    dynesty_chain[:,i]= cdf[:]*sampx[:]
    


