# %%

oph = 0

if oph:
    # OPH

    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("--sampler", default='emcee', choices=["emcee", "zeus", "dynesty"])
    parser.add_argument("--user", required=True, choices=["MT", "NB", "MM", "NP"])
    parser.add_argument("--run", required=True)
    parser.add_argument("--append", default=False)
    args = parser.parse_args()
    user = args.user
    run, chain_letter = int(args.run[0]), args.run[1]
    restart = bool(int(args.append))
    subscript_to_restart = str(run)+chain_letter
    sampler = args.sampler

    print(user, run, chain_letter, restart, subscript_to_restart, sampler)
    print(type(user), type(run), type(chain_letter), type(restart), type(subscript_to_restart), type(sampler))

    dir_home       = "/home/STUDENTI/niccolo.passaleva/Tesi"
    dir_CHIMERA    = dir_home+"/CHIMERA-main/"
    dir_data       = dir_home+"/data/"
    dir_parent     = dir_data+"MICEv2_1.6M_L-compl_UCV.h5"
    dir_parent_int = dir_data+"p_bkg_gauss_smooth_zerr0.001_zres5000_smooth30.pkl"
    dir_out        = "/home/STUDENTI/niccolo.passaleva/Tesi/outputs/" 



else:
    # Local
    dir_CHIMERA  = "/opt/CHIMERA-main/"
    dir_data     = "/opt/CHIMERA-main/data/"
    dir_parent     = dir_data+"MICEv2_1.6M_L-compl_UCV.h5"
    dir_parent_int = dir_data+"p_bkg_gauss_smooth_zerr0.001_zres5000_smooth30.pkl"
    dir_out        = "/Users/niccolopassaleva/Desktop/Tesi/codici/chains/"

    user, run, chain_letter, restart, subscript_to_restart, sampler = "NP", 4, "a", False, "1a", "dynesty"

# Recipe 
recipe  = {"Nevents" :          None,
           "Nsamples" :         5000,
           "data_GAL_zerr":     0.001,
           "data_GW_smooth":    0.3,
           "npix_event":        15,
           "sky_conf":          0.90,
           "nside_list":        [512,256,128,64,32,16,8],
           "z_int_sigma":       5,
           "z_int_res":         500,
           "z_det_range":       [0.01, 1.3],
           "H0_prior":          [30.,200.],
           "nsteps":            1000,
           "todo_mock":         "O4_v3",
           "data_GAL_dir":      dir_parent,
           "data_GAL_int_dir":  dir_parent_int,
           "neff_min":          5,
}

# Flat priors
priors_c = {}
priors_m = {"lambda_peak":      [0.01,0.99],
            "alpha":            [1.5,12.],
            "beta":             [-4,12.],
            "delta_m":          [0.01,10.],
            "ml":               [2.,50.],
            "mh":               [50.,200.],
            "mu_g":             [2.,50.,],
            "sigma_g":          [0.4,10.],}
priors_r =  {} 
priors = {}
[priors.update(d) for d in [priors_c, priors_m, priors_r]]

params_keys = priors.keys()

# %%
# IMPORTS & FUNCTIONS

import os, sys, logging, json, re, h5py

os.environ["HDF5_USE_FILE_LOCKING"] = "FALSE"
os.environ["OMP_NUM_THREADS"] = "1"

import warnings
warnings.filterwarnings("ignore")

import numpy as np
import h5py
from IPython.display import display, Math
from chainconsumer import ChainConsumer
import time

if sampler == 'zeus':
    import zeus
    from zeus import ChainManager
    from multiprocessing import Pool

elif sampler == 'emcee':
    from schwimmbad import MPIPool
    import emcee

elif sampler == 'dynesty':
    from schwimmbad import MPIPool
    import dynesty
    from dynesty import plotting as dyplot
    from dynesty import utils as dyu

sys.path.append(dir_CHIMERA)
from CHIMERA.Bias import Bias
from CHIMERA.astro.rate import phi_MD
from CHIMERA.astro.mass import pdf_PLP
from CHIMERA.cosmo import fLCDM
from CHIMERA.Likelihood import MockLike
from CHIMERA.DataGW import DataGWMock
from CHIMERA.utils import presets

# %%
# LOADING DATA

if recipe["todo_mock"]=="O4_v3":

    dir_GW_data         = dir_data + "O4_v3/samples_from_fisher_allpars_snrth-12_ieth-0.05_DelOmTh-inf.h5"
    dataset_GW          = DataGWMock(dir_GW_data)
    data_GW             = dataset_GW.load(Nsamples=recipe["Nsamples"], keys_load=["m1det", "m2det", "phi", "dL", "theta"])
    Nevents             = dataset_GW.Nevents

    dir_Inj_data        = dir_data + "O4_v2/injections_40M_sources_PLP_v9_H1-L1-Virgo-KAGRA_IMRPhenomHM_snr_th-12_dutyfac-1_fmin-10_noiseless.h5"

    recipe["file_inj"]  = dir_Inj_data
    recipe["N_inj"]     = 40*10**6
    recipe["SNRth_inj"] = 12

if recipe["todo_mock"]=="O5_v3":

    dir_GW_data         = dir_data + "O5_v3/O5_samples_from_fisher_allpars_snrth-25_ieth-0.05_DelOmTh-inf.h5"
    dataset_GW          = DataGWMock(dir_GW_data)
    data_GW             = dataset_GW.load(Nsamples=recipe["Nsamples"], keys_load=["m1det", "m2det", "phi", "dL", "theta"])
    Nevents             = dataset_GW.Nevents

    dir_Inj_data        = dir_data + "O5_v3/injections_20M_sources_PLP_v9s2_H1-L1-Virgo-KAGRA-LIGOI_IMRPhenomHM_snr_th-20_dutyfac-1_fmin-10_noiseless.h5"

    recipe["file_inj"]  = dir_Inj_data
    recipe["N_inj"]     = 40*10**6
    recipe["SNRth_inj"] = 12

data_GW_names      = ["Mock_{:02d}".format(i) for i in range(Nevents)]

recipe["Nevents"]  = Nevents
recipe["ndim"]     = len(params_keys)
recipe["nwalkers"] = 2*recipe["ndim"]

lambda_mass_true  = presets.lambda_mass_PLP_mock_v1
lambda_cosmo_true = presets.lambda_cosmo_mock_v1
lambda_rate_true  = presets.lambda_rate_Madau_mock_v2

lambda_true = {}
[lambda_true.update(d) for d in [lambda_cosmo_true, lambda_mass_true, lambda_rate_true]];

if not oph:
    logging.basicConfig(format="%(asctime)s - %(levelname)s - %(message)s", filename=dir_out+'log.log', level=logging.INFO)
    log = logging.getLogger(__name__)

# %%
# INITIALIZE LIKELIHOOD & BIAS

like = MockLike(model_cosmo = fLCDM, 
                model_mass  = pdf_PLP, 
                model_rate  = phi_MD,
    
                # Data
                data_GW          = data_GW,
                data_GW_names    = data_GW_names,
                data_GW_smooth   = recipe["data_GW_smooth"],
                data_GAL_dir     = recipe["data_GAL_dir"],
                data_GAL_int_dir = recipe["data_GAL_int_dir"],
                data_GAL_zerr    = recipe["data_GAL_zerr"],
    
                # Parameters for pixelization
                nside_list       = recipe["nside_list"], 
                npix_event       = recipe["npix_event"],
                sky_conf         = recipe["sky_conf"],
    
                # Parameters for integration in redshift
                z_int_H0_prior   = recipe["H0_prior"],
                z_int_sigma      = recipe["z_int_sigma"],
                z_int_res        = recipe["z_int_res"],
                z_det_range      = recipe["z_det_range"],

                # Check Neff
                neff_data_min    = recipe["neff_min"],
                )
# %%
bias = Bias(model_cosmo = like.gw.model_cosmo, model_mass = like.gw.model_mass, model_rate = like.gw.model_rate,
            
            # Backgroung galaxies' distribution
            p_bkg         = like.p_gal_bkg,

            # Injections file
            file_inj      = recipe["file_inj"], 
            N_inj         = recipe["N_inj"], 

            # SNR and z_range
            snr_th        = recipe["SNRth_inj"], 
            z_det_range   = recipe["z_det_range"],

            # Check Neff
            neff_inj_min = recipe["neff_min"],
            )

# %%
# Save likelihood object properties
if (user=="NB") & (run==1):
    like.save(dir_out+"like_properties.pkl")

# %%
# Analysis
bests      = np.array([lambda_true[x] for x in params_keys])
hyperspace = np.array([priors[x] for x in params_keys])
sigmas     = 0.1 * np.diff(hyperspace).flatten()

def log_prior(lambdas):
    if all(hyperspace[:, 0] <= lambdas) and all(hyperspace[:, 1] >= lambdas):
        return 0
    return -np.inf

def log_prior_nested(u):
    v = u * 1
    for i in range(len(u)):
        v[i]=(hyperspace[i,1]-hyperspace[i,0])*u[i]+hyperspace[i,0]
    return v

def check_initials(initial_values):
     for i in range(initial_values.shape[0]):
         if log_prior(initial_values[i,:])==-np.inf:
             return False
     return True

def log_prob_mass_cosmo(lambdas):
    
    lp = log_prior(lambdas)

    if not np.isfinite(lp):
        #print("Step with -inf prob :(")
        return -np.inf
    
    parameters   = dict(zip(params_keys, lambdas))
    lambda_cosmo = lambda_cosmo_true
    lambda_mass  = {k : parameters[k] for k in priors_m.keys()}
    lambda_rate  = lambda_rate_true

    likes = like.compute(lambda_cosmo=lambda_cosmo, lambda_mass=lambda_mass, lambda_rate=lambda_rate)
    beta  = bias.compute(lambda_cosmo=lambda_cosmo, lambda_mass=lambda_mass, lambda_rate=lambda_rate)

    llike = np.log(likes)
    lbeta = np.log(beta)

    finite  = np.isfinite(llike)
    llike[~finite] = -1000

    res = np.sum(llike) - np.sum(finite)*lbeta

    return res

def lnlike(lambdas):
    
    parameters   = dict(zip(params_keys, lambdas))
    lambda_cosmo = lambda_cosmo_true
    lambda_mass  = {k : parameters[k] for k in priors_m.keys()}
    lambda_rate  = lambda_rate_true

    likes = like.compute(lambda_cosmo=lambda_cosmo, lambda_mass=lambda_mass, lambda_rate=lambda_rate)
    beta  = bias.compute(lambda_cosmo=lambda_cosmo, lambda_mass=lambda_mass, lambda_rate=lambda_rate)

    llike = np.log(likes)
    lbeta = np.log(beta)

    finite  = np.isfinite(llike)
    llike[~finite] = -1000

    ll = np.sum(llike) - np.sum(finite)*lbeta

    return ll

def generate_chain_filename(user, run, letter, restart = restart, subscript_to_restart = subscript_to_restart):

    if recipe["todo_mock"]=="O5_v3":
        if not restart:
            filename = f'chain_{user}_{str(run-1)}{letter}_CosmoRateFixed_O5.h5'
            
        else:
            run_to_restart    = ''.join(filter(str.isdigit, subscript_to_restart))
            letter_to_restart = ''.join(filter(str.isalpha, subscript_to_restart))
            filename = f'chain_{user}_{str(int(run_to_restart)+1)}{letter_to_restart}_CosmoRateFixed_O5.h5'

    if recipe["todo_mock"]=="O4_v3":
        if not restart:
            filename = f'chain_{user}_{str(run-1)}{letter}_CosmoRateFixed_O4.h5'
            
        else:
            run_to_restart    = ''.join(filter(str.isdigit, subscript_to_restart))
            letter_to_restart = ''.join(filter(str.isalpha, subscript_to_restart))
            filename = f'chain_{user}_{str(int(run_to_restart)+1)}{letter_to_restart}_CosmoRateFixed_O4.h5'
 
    return filename

def generate_plots_filename(user, run, letter):

    if recipe["todo_mock"]=="O5_v3":
        plotname = f'chain_{user}_{str(run)}{letter}_CosmoRateFixed_O5'

    if recipe["todo_mock"]=="O4_v3":
        plotname = f'chain_{user}_{str(run)}{letter}_CosmoRateFixed_O4'

    return plotname

def generate_dynestysave_name(user, run, letter):
    
    if recipe["todo_mock"]=="O5_v3":
        dynestysave = f'dynesty_{user}_{str(run)}{letter}_CosmoRateFixed_O5.save'

    if recipe["todo_mock"]=="O4_v3":
        dynestysave = f'dynesty_{user}_{str(run)}{letter}_CosmoRateFixed_O4.save'

    return dynestysave

def last_chain_point(chain_name, path = ''):
    try:
        with h5py.File(path+chain_name, 'r') as file:
            if sampler == 'zeus':
                dataset = file['samples'][:]  # Replace 'dataset_name' with the actual name of your dataset
            elif sampler == 'emcee':
               dataset = file['mcmc']['chain'][:]
            
            rows_with_all_zeros = np.all(dataset == 0, axis=(1,2)) 
            filtered_dataset = dataset[~rows_with_all_zeros]
            last_row = filtered_dataset[-1]  # Assuming rows are stored as separate entries in the dataset
            return last_row

    except (IOError, KeyError):
        print(f"Error opening file {chr(chain_name)} or dataset not found. Return None.")
        return None

def generate_initial_conditions( nwalkers,
                                 ndim,
                                 distribution='gaussian',
                                 priors=None,
                                 gaussian_bests=None,
                                 gaussian_sigmas=None,
                                 restart = restart,
                                 subscript_to_restart = subscript_to_restart):

    if not restart:

        if priors is None:
            priors = np.tile([-np.inf, np.inf])

        if gaussian_bests is None:
            gaussian_bests = np.ones(ndim)

        if gaussian_sigmas is None:
            gaussian_sigmas = np.full(ndim, 0.2)

        start = np.zeros((nwalkers, ndim))

        if distribution == 'gaussian':
            for i in range(nwalkers):
                tmp = np.random.normal(loc=gaussian_bests, scale=gaussian_sigmas, size=(1, ndim))
                while not check_initials(tmp):
                    tmp = np.random.normal(loc=gaussian_bests, scale=gaussian_sigmas, size=(1, ndim))
                start[i] = tmp
            
        elif distribution == 'truncgauss':
            start  = np.random.normal(loc=bests, scale=sigmas, size=(nwalkers, ndim))
            outside_indices = np.logical_or(start < hyperspace[:, 0], start > hyperspace[:, 1])
            for i in range(ndim):
                start[outside_indices[:, i], i] = np.random.uniform(low=hyperspace[i, 0], high=hyperspace[i, 1], size=np.sum(outside_indices[:, i]))
            
        elif distribution == 'uniform':
            for i in range(nwalkers):
                tmp = np.random.uniform(low=priors[:, 0], high=priors[:, 1], size=(1, ndim))
                while not check_initials(tmp):
                    tmp = np.random.uniform(low=priors[:, 0], high=priors[:, 1], size=(1, ndim))
                    while not check_initials(tmp):
                        tmp = np.random.uniform(low=priors[:, 0], high=priors[:, 1], size=(1, ndim))
                    start[i] = tmp
        else:
            print("Only admitted distributions are 'gaussian', 'uniform', and 'truncgauss'.")
            return None
        
        return start

    else: # start from last point
        chain_to_restart = f'chain_{user}_{subscript_to_restart}.h5'
        start = last_chain_point(chain_to_restart, path = dir_out)
        return start

#####################################################
#                                                   # 
#                      MCMC                         #
#                                                   #
#####################################################

filename = generate_chain_filename(user, run, chain_letter, restart = restart, subscript_to_restart = subscript_to_restart)
plotname = generate_plots_filename(user, run, chain_letter)

if sampler == 'dynesty':

    dynestysave = generate_dynestysave_name(user, run, chain_letter)

if sampler != 'dynesty':
    
    initial_values = generate_initial_conditions(recipe['nwalkers'], 
                                                recipe['ndim'], 
                                                distribution = 'truncgauss',
                                                priors = hyperspace,
                                                gaussian_bests = bests,
                                                gaussian_sigmas = sigmas,
                                                restart = restart,
                                                subscript_to_restart = subscript_to_restart)

ndim = recipe["ndim"]

def sampling(sampler, truths, nwalkers=5*ndim, nsteps=1000, quantiles_value=[16,50,84], running_time=True, plot=False):
    
    quantiles_list = list(np.array(quantiles_value)*0.01)
    c = ChainConsumer()
    
    #Routine for the 'emcee' sampler
    if "emcee" in sampler: 
        
        thin = 1
        burnin = 300
        
        if os.path.isfile(dir_out+filename) and not restart:
            reader = emcee.backends.HDFBackend(dir_out+filename)
            flat_samples = reader.get_chain(discard=burnin, flat=True, thin=thin)
        
        else:
            p0 = initial_values #Initial position of walkers
            
            backend = emcee.backends.HDFBackend(dir_out+filename)
            backend.reset(nwalkers, ndim)

            start = time.time()
        
            if oph:
              with MPIPool() as pool:
                  if not pool.is_master():
                      pool.wait()
                      sys.exit(0)
               
                  esampler = emcee.EnsembleSampler(nwalkers, ndim, log_prob_mass_cosmo, backend=backend, pool=pool)
                  esampler.run_mcmc(p0, nsteps)
        
            else:
                esampler = emcee.EnsembleSampler(nwalkers, ndim, log_prob_mass_cosmo, backend=backend)
                esampler.run_mcmc(p0, nsteps, progress=True)
            
        
            flat_samples = esampler.get_chain(discard=burnin, thin=thin, flat=True)
            end = time.time()
        
        if (running_time):
               sampling_time = end - start
               print("Sampling time of emcee (s):", sampling_time)
        
        #Results printing
        quantiles = np.empty([ndim, 3])
        fit_values = np.empty(ndim)
        
        print("emcee parameter estimation:")
        for i in range(ndim):
            quantiles[i][:] = np.percentile(flat_samples[:, i], quantiles_value)
            fit_values[i] =  quantiles[i][1]
            txt = "\mathrm{{{3}}} = {0:.3f}_{{-{1:.3f}}}^{{{2:.3f}}}"
            txt = txt.format(fit_values[i], fit_values[i]-quantiles[i][0], fit_values[i]-quantiles[i][2], list(params_keys)[i])
            display(Math(txt))
        
        c.add_chain(flat_samples, parameters=list(params_keys), name='emcee')
   
    #Routine for the 'zeus' sampler
    if sampler == "zeus":

        thin = 1
        burnin = 300
              
        if os.path.isfile(dir_out+filename) and not restart:
    
            with h5py.File(dir_out+filename, "r") as hf:
                flat_samples = np.copy(hf['flat_samples'])
        
        else:
            p0 = initial_values 

            start = time.time()
            
            if oph:
               with ChainManager(1) as cm:                       
                   zsampler = zeus.EnsembleSampler(nwalkers, ndim, log_prob_mass_cosmo, pool=cm.getpool)
                   zsampler.run_mcmc(p0, 5000, callbacks=zeus.callbacks.SaveProgressCallback(dir_out+filename, ncheck=10))
            
            else:
                zsampler = zeus.EnsembleSampler(nwalkers, ndim, log_prob_mass_cosmo)
                zsampler.run_mcmc(p0, 5000, callbacks=zeus.callbacks.SaveProgressCallback(dir_out+filename, ncheck=10))
            

        
            flat_samples = zsampler.get_chain(discard=burnin, thin=thin, flat=True)
            end = time.time()
            
            hf = h5py.File(dir_out+filename, 'w')
            hf.create_dataset('flat_samples', data=flat_samples)
            hf.close()

            if (running_time):
                sampling_time = end - start
                print("Sampling time of zeus (s):", sampling_time)
        
        quantiles = np.empty([ndim, 3])
        fit_values = np.empty(ndim)
        
        print("zeus parameter estimation:")
        for i in range(ndim):
            quantiles[i][:] = np.percentile(flat_samples[:, i], quantiles_value)
            fit_values[i] =  quantiles[i][1]
            txt = "\mathrm{{{3}}} = {0:.3f}_{{-{1:.3f}}}^{{{2:.3f}}}"
            txt = txt.format(fit_values[i], fit_values[i]-quantiles[i][0], fit_values[i]-quantiles[i][2], list(params_keys)[i])
            display(Math(txt))
        
        c.add_chain(flat_samples, parameters=list(params_keys), name='zeus')
             
    #Routine for the 'dynesty' sampler
    if "dynesty" in sampler:

        if os.path.isfile(filename) and not restart:
            with h5py.File(filename, "r") as hf:
                flat_samples = np.copy(hf['flat_samples'])
                importance_weights = np.copy(hf['weights'])
                evidence = np.copy(hf['evidence'])

        else:
            if os.path.isfile(dir_out+dynestysave) and restart:
                start = time.time()
                
                if oph:
                    with MPIPool() as pool:
                        if not pool.is_master():
                            pool.wait()
                            sys.exit(0)
                            
                        dsampler = dynesty.NestedSampler.restore(dir_out+dynestysave, pool=pool)
                        dsampler.run_nested(resume=True,checkpoint_file=dir_out+dynestysave)
                
                else:
                    dsampler = dynesty.NestedSampler.restore(dir_out+dynestysave)
                    dsampler.run_nested(resume=True,checkpoint_file=dir_out+dynestysave)
                    

            else:
                rstate = np.random.default_rng(56101)
                start = time.time()
                
                if oph:
                    with MPIPool() as pool:
                        if not pool.is_master():
                            pool.wait()
                            sys.exit(0)
                            
                        dsampler = dynesty.NestedSampler(lnlike, log_prior_nested, ndim=ndim, 
                                                    bound='multi', sample='unif', rstate=rstate, pool=pool)                            
                        dsampler.run_nested(checkpoint_file=dir_out+dynestysave)
                    
                else:
                   dsampler = dynesty.NestedSampler(lnlike, log_prior_nested, ndim=ndim, 
                                                    bound='multi', sample='unif', rstate=rstate)                            
                   dsampler.run_nested(checkpoint_file=dir_out+dynestysave) 
                
            dres = dsampler.results
            end = time.time()
            flat_samples = dres.samples
            importance_weights = dres.importance_weights()
            evidence = dres.logz
                
            hf = h5py.File(dir_out+filename, 'w')
            hf.create_dataset('flat_samples', data=dres.samples)
            hf.create_dataset('weights', data=importance_weights)
            hf.create_dataset('evidence', data=evidence)
            hf.close()
                
            if (running_time):
                sampling_time = end - start
                print("Sampling time of dynesty (s):", sampling_time)
        
        quantiles = np.empty([ndim, 3])
        fit_values = np.empty(ndim)

        print("dynesty parameter estimation:")
        for i in range(ndim):
            q=np.array(quantiles_list)
            sampx = np.atleast_1d(flat_samples[:,i])
            weights = importance_weights
            weights = np.atleast_1d(weights)
            idx = np.argsort(sampx)  # sort samples
            sw = weights[idx]  # sort weights
            cdf = np.cumsum(sw)[:-1]  # compute CDF
            cdf /= cdf[-1]  # normalize CDF
            cdf = np.append(0, cdf)  # ensure proper span
            quantiles[i][:] = np.interp(q, cdf, sampx[idx]).tolist()
            fit_values[i] = quantiles[i][1]
            txt = "\mathrm{{{3}}} = {0:.3f}_{{-{1:.3f}}}^{{{2:.3f}}}"
            txt = txt.format(fit_values[i], fit_values[i]-quantiles[i][0], fit_values[i]-quantiles[i][2], list(params_keys)[i])
            display(Math(txt))
        
        print("The evidence is:",evidence[-1])
            
        c.add_chain(flat_samples, parameters=list(params_keys), weights=importance_weights, name='dynesty')
    
    c.configure(legend_artists=True, sigma2d=True)
    fig = c.plotter.plot(filename=plotname+'_corner.png',truth=truths)
    fig1 = c.plotter.plot_walks(filename=plotname+'_walks.png',truth=truths, convolve=150)
    fig2 = c.plotter.plot_distributions(filename=plotname+'_distrib.png',truth=truths)
    fig.set_size_inches(3 + fig.get_size_inches())
    fig1.set_size_inches(3 + fig1.get_size_inches())
    fig2.set_size_inches(3 + fig2.get_size_inches())
        

sampling(sampler, lambda_mass_true, recipe["nwalkers"])

