import warnings
import os
from sklearn.exceptions import InconsistentVersionWarning
warnings.filterwarnings("ignore", category=InconsistentVersionWarning)
warnings.filterwarnings(
    "ignore",
    message=".*column is deprecated.*",
    module=r"sacc\.sacc"
)
warnings.filterwarnings(
    "ignore",
    category=RuntimeWarning,
    message=r".*invalid value encountered*"
)
warnings.filterwarnings(
    "ignore",
    category=RuntimeWarning,
    message=r".*overflow encountered*"
)
warnings.filterwarnings(
    "ignore",
    category=UserWarning,
    message=r".*Function not smooth or differentiabl*"
)
warnings.filterwarnings(
    "ignore",
    category=UserWarning,
    message=r".*Hartlap correction*"
)
import functools, iminuit, copy, argparse, random, time 
import emcee, itertools
import numpy as np
from cobaya.yaml import yaml_load
from cobaya.model import get_model
from getdist import IniFile
from schwimmbad import MPIPool
# ------------------------------------------------------------------------------
# ------------------------------------------------------------------------------
# ------------------------------------------------------------------------------
# ------------------------------------------------------------------------------
# ------------------------------------------------------------------------------
# ------------------------------------------------------------------------------
_affinity_set = False

def enforce_affinity():
    local_rank = int(os.environ.get("OMPI_COMM_WORLD_LOCAL_RANK",
                     os.environ.get("MPI_LOCALRANKID", 0)))
    omp_threads = int(os.environ.get("OMP_NUM_THREADS", 1))
    first_core = local_rank * omp_threads
    last_core  = first_core + omp_threads - 1
    try:
        p = psutil.Process()
        p.cpu_affinity(list(range(first_core, last_core + 1)))
    except Exception as e:
        rank = int(os.environ.get("OMPI_COMM_WORLD_RANK", 0))
        print(f"[Rank {rank}] Failed to set affinity: {e}")
# ------------------------------------------------------------------------------
# ------------------------------------------------------------------------------
# ------------------------------------------------------------------------------
# ------------------------------------------------------------------------------
# ------------------------------------------------------------------------------
# ------------------------------------------------------------------------------
parser = argparse.ArgumentParser(prog='EXAMPLE_EMUL_PROFILE1')
parser.add_argument("--nstw",
                    dest="nstw",
                    help="Number of likelihood evaluations (steps) per temperature per walker",
                    type=int,
                    nargs='?',
                    const=1,
                    default=200)
parser.add_argument("--root",
                    dest="root",
                    help="Name of the Output File",
                    nargs='?',
                    const=1,
                    default="./projects/example/")
parser.add_argument("--outroot",
                    dest="outroot",
                    help="Name of the Output File",
                    nargs='?',
                    const=1,
                    default="test.dat")
parser.add_argument("--profile",
                    dest="profile",
                    help="Which Parameter to Profile",
                    type=int,
                    nargs='?',
                    const=1,
                    default=1)
parser.add_argument("--factor",
                    dest="factor",
                    help="Factor that set the bounds (multiple of cov matrix)",
                    type=float,
                    nargs='?',
                    const=1.0,
                    default=3.0)
parser.add_argument("--numpts",
                    dest="numpts",
                    help="Number of Points to Compute Minimum",
                    type=int,
                    nargs='?',
                    const=1,
                    default=20)
parser.add_argument("--minfile",
                    dest="minfile",
                    help="Minimization Result",
                    nargs='?',
                    const=1)
parser.add_argument("--cov",
                    dest="cov",
                    help="Chain Covariance Matrix",
                    nargs='?',
                    const=1,
                    default=None)
args, unknown = parser.parse_known_args()
# ------------------------------------------------------------------------------
# ------------------------------------------------------------------------------
# ------------------------------------------------------------------------------
# ------------------------------------------------------------------------------
# ------------------------------------------------------------------------------
# ------------------------------------------------------------------------------
yaml_string = r"""
likelihood:
  planck_2018_highl_plik.TTTEEE_lite:
    path: ./external_modules/
    clik_file: plc_3.0/hi_l/plik_lite/plik_lite_v22_TTTEEE.clik
  planck_2018_lowl.TT:
    path: ./external_modules
  planck_2018_lowl.EE:
    path: ./external_modules
  sn.desy5:
  bao.desi_dr2.desi_bao_all:
  act_dr6_lenslike.ACTDR6LensLike:
    lens_only: false
    variant: actplanck_baseline

params:
  logA:
    prior:
      min: 2.8
      max: 3.2
    ref:
      dist: norm
      loc: 3.04
      scale: 0.025
    proposal: 0.025
    latex: \log(10^{10} A_\mathrm{s})
    drop: true
  As:
    value: 'lambda logA: 1e-10*np.exp(logA)'
    latex: A_\mathrm{s}
  ns:
    prior:
      min: 0.93
      max: 1.01
    ref:
      dist: norm
      loc: 0.96
      scale: 0.0075
    proposal: 0.0075
    latex: n_\mathrm{s}
  thetastar100:
    prior:
      min: 1
      max: 1.2
    ref:
      dist: norm
      loc: 1.041
      scale: 0.001
    proposal: 0.001
    latex: 100\theta_\mathrm{*}
    renames: theta
    drop: true
  thetastar:
    value: 'lambda thetastar100: 1.e-2*thetastar100'
    latex: \theta_\mathrm{*}
  omegabh2:
    prior:
      min: 0.01
      max: 0.03
    ref:
      dist: norm
      loc: 0.022383
      scale: 0.005
    proposal: 0.005
    latex: \Omega_\mathrm{b} h^2
  omegach2:
    prior:
      min: 0.08
      max: 0.16
    ref:
      dist: norm
      loc: 0.12011
      scale: 0.01
    proposal: 0.01
    latex: \Omega_\mathrm{c} h^2
  tau:
    prior:
      min: 0.04
      max: 0.1
    ref:
      dist: norm
      loc: 0.055
      scale: 0.01
    proposal: 0.01
    latex: \tau_\mathrm{reio}
  omegaaxh2:
    prior:
      min: 0.0001
      max: 0.4
    ref:
      dist: norm
      loc: 0.1
      scale: 0.03
    proposal: 0.03
    latex: \Omega_\mathrm{ax} h^2
    drop: true
  omaxh2:
    value: 'lambda omegaaxh2: omegaaxh2'
    latex: \omega_\mathrm{ax}
  logmx:
    prior:
      min: -34
      max: -31.0
    ref:
      dist: norm
      loc: -33
      scale: 0.07
    proposal: 0.07
    latex: \log{m_{\rm ax}}
    drop: true
  m_ax:
    value: 'lambda logmx: 10**logmx'
    latex: m_\mathrm{ax}
  mnu:
    value: 0.06
  H0:
    derived: true
    latex: H_0
  omegam:
    derived: true
    latex: \Omega_\mathrm{m}
  rdrag:
    derived: true
    latex: r_\mathrm{drag}

theory:
  camb:
    path: ./external_modules/code/axiecamb
    use_renames: true
    extra_args:
      num_massive_neutrinos: 1
      nnu: 3.046
      theta_H0_range: [40, 130]
      halofit_version: original
      lens_potential_accuracy: 4
      lens_margin: 1250
      AccuracyBoost: 1.0
      lSampleBoost: 1.0
      lAccuracyBoost: 1.0  
"""
# ------------------------------------------------------------------------------
# ------------------------------------------------------------------------------
# ------------------------------------------------------------------------------
# ------------------------------------------------------------------------------
# ------------------------------------------------------------------------------
# ------------------------------------------------------------------------------
model = get_model(yaml_load(yaml_string))
def chi2(p):
    p = [float(v) for v in p.values()] if isinstance(p, dict) else p
    if np.any(np.isinf(p)) or  np.any(np.isnan(p)):
      raise ValueError(f"At least one parameter value was infinite (CoCoa) param = {p}")
    point = dict(zip(model.parameterization.sampled_params(), p))
    res1 = model.logprior(point,make_finite=False)
    if np.isinf(res1) or  np.any(np.isnan(res1)):
      return 1e20
    res2 = model.loglike(point,
                         make_finite=False,
                         cached=False,
                         return_derived=False)
    if np.isinf(res2) or  np.any(np.isnan(res2)):
      return 1e20
    return -2.0*(res1+res2)
def chi2v2(p):
    p = [float(v) for v in p.values()] if isinstance(p, dict) else p
    point = dict(zip(model.parameterization.sampled_params(), p))
    logposterior = model.logposterior(point, as_dict=True)
    chi2likes=-2*np.array(list(logposterior["loglikes"].values()))
    chi2prior=-2*np.atleast_1d(model.logprior(point,make_finite=False))
    return np.concatenate((chi2likes, chi2prior))
# ------------------------------------------------------------------------------
# ------------------------------------------------------------------------------
# ------------------------------------------------------------------------------
# ------------------------------------------------------------------------------
# ------------------------------------------------------------------------------
# ------------------------------------------------------------------------------
def min_chi2(x0,
             cov, 
             fixed=-1, 
             nstw=200,
             nwalkers=5,
             pool=None):

    def mychi2(params, *args):
        z, fixed, T = args
        params = np.array(params, dtype='float64')
        if fixed > -1:
            params = np.insert(params, fixed, z)
        return chi2(p=params)/T
    if fixed > -1:
        z      = x0[fixed]
        x0     = np.delete(x0, (fixed))
        args = (z, fixed, 1.0)
        cov = np.delete(cov, (fixed), axis=0)
        cov = np.delete(cov, (fixed), axis=1)
    else:
        args = (0.0, -2.0, 1.0)

    def logprob(params, *args):
        res = mychi2(params, *args)
        if (res > 1.e19 or np.isinf(res) or  np.isnan(res)):
          return -np.inf
        else:
          return -0.5*res
    
    class GaussianStep:
       def __init__(self, stepsize=0.2):
           self.cov = stepsize*cov
       def __call__(self, x):
           return np.random.multivariate_normal(x, self.cov, size=1)
    
    ndim        = int(x0.shape[0])
    nwalkers    = int(nwalkers)
    nstw        = int(nstw)
    if fixed == -1:
      temperature = np.array([1.0, 0.25, 0.1, 0.005, 0.001], dtype='float64')
    else:
      temperature = np.array([0.3, 0.1, 0.005, 0.001], dtype='float64')
    stepsz      = temperature/3.0

    partial_samples = [x0]
    partial = [mychi2(x0, *args)]

    for i in range(len(temperature)):
        x = [] # Initial point
        for j in range(nwalkers):
            x.append(GaussianStep(stepsize=stepsz[i])(x0)[0,:]) 
        sampler = emcee.EnsembleSampler(nwalkers=nwalkers, 
                                        ndim=ndim, 
                                        log_prob_fn=logprob, 
                                        args=(args[0], args[1], temperature[i]),
                                        moves=[(emcee.moves.DEMove(), 0.8),
                                               (emcee.moves.DESnookerMove(), 0.2)],
                                        pool=pool)
        sampler.run_mcmc(np.array(x,dtype='float64'), 
                         nstw, 
                         skip_initial_state_check=True)
        samples = sampler.get_chain(flat=True, discard=0)
        j = np.argmin(-1.0*np.array(sampler.get_log_prob(flat=True)))
        partial_samples.append(samples[j])
        partial.append(mychi2(samples[j], *args))
        x0 = copy.deepcopy(samples[j])
        sampler.reset()
    # min chi2 from the entire emcee runs
    j = np.argmin(np.array(partial))
    return partial_samples[j]
# ------------------------------------------------------------------------------
# ------------------------------------------------------------------------------
# ------------------------------------------------------------------------------
# ------------------------------------------------------------------------------
# ------------------------------------------------------------------------------
# ------------------------------------------------------------------------------
def prf(x0, nstw, cov, fixed=-1, nwalkers=5, pool=None):
    res =  min_chi2(x0=np.array(x0, dtype='float64'), 
                    fixed=fixed,
                    cov=cov, 
                    nstw=nstw, 
                    nwalkers=nwalkers,
                    pool=pool)
    return res
# ------------------------------------------------------------------------------
# ------------------------------------------------------------------------------
# ------------------------------------------------------------------------------
# ------------------------------------------------------------------------------
# ------------------------------------------------------------------------------
# ------------------------------------------------------------------------------
if __name__ == '__main__':
    enforce_affinity() # enforce affinity (so Hybrid MPI-OpenMP works)!
    with MPIPool() as pool:
        if not pool.is_master():
            pool.wait()
            sys.exit(0)
        dim      = model.prior.d()     
        nwalkers = max(3*dim, pool.comm.Get_size())
        nstw = args.nstw

        # 1st: load the cov. matrix --------------------------------------------
        if args.cov is None:
          cov = model.prior.covmat(ignore_external=False) # cov from prior
          factor = min(1.0, args.factor)
        else:
          cov = np.loadtxt(args.root+args.cov)[0:model.prior.d(),0:model.prior.d()]
          factor = args.factor
        sigma = np.sqrt(np.diag(cov))

        # 2nd: Get minimum --------------------------------------------------
        if args.minfile is not None: # load minimum from running MCMC
          x0 = np.loadtxt(args.minfile)
          chi20 = x0[-1]
          x0 = x0[0:model.prior.d()]
        else: # Compute the minimum (slow)
          (x0, results) = model.get_valid_point(max_tries=1000, 
                                     ignore_fixed_ref=False,
                                     logposterior_as_dict=True)
          res = np.array(list(prf(x0=x0, 
                                  nstw=int(5.*nstw/4.), 
                                  nwalkers=nwalkers,
                                  pool=pool,
                                  cov=cov,
                                  fixed=-1)), dtype="object")
          x0 = np.array(res, dtype='float64')[0:model.prior.d()]
          chi20 = chi2(x0)
          print(f"Global Min: params = {x0}, and chi2 = {chi20}")

        # Test consistency of the min and profile codes
        if (abs(chi2(x0)-chi20)>0.02):
          raise ValueError("Inconsistency Min and Profile setups")

        # 3rd: Set the parameter profile range ---------------------------------
        start = np.zeros(model.prior.d(), dtype='float64')
        stop  = np.zeros(model.prior.d(), dtype='float64')
        start = x0 - factor*sigma
        stop  = x0 + factor*sigma
        
        # We need to respect the YAML priors
        bounds0 = model.prior.bounds(confidence=0.999999)
        for i in range(model.prior.d()):
            if (start[i] < bounds0[i][0]):
              start[i] = bounds0[i][0]
            if (stop[i] > bounds0[i][1]):
              stop[i] = bounds0[i][1]

        half_range = (stop[args.profile] - start[args.profile]) / 2.0
       
        numpts = args.numpts-1 if args.numpts%2 == 1 else args.numpts 
      
        param  = np.linspace(start = x0[args.profile] - half_range,
                             stop  = x0[args.profile] + half_range,
                             num = numpts)
        numpts=numpts+1
        param = np.insert(param, numpts//2, x0[args.profile])
        
        # 4th Print to the terminal ---------------------------------------------
        names = list(model.parameterization.sampled_params().keys()) # Cobaya Call
        print(f"nstw (evals/Temp/walkers)={args.nstw}, "
              f" param={names[args.profile]}\n"
              f"profile param values = {param}")
        
        # 5th: Set the vectors that will hold the final result -----------------
        xf = np.tile(x0, (numpts, 1))
        xf[:,args.profile] = param

        chi2res = np.zeros(numpts)  
        chi2res[numpts//2] = chi20
        
        # 5th: run from midpoint to right --------------------------------------
        tmp = np.array(xf[numpts//2,:], dtype='float64')
        for i in range(numpts//2+1,numpts): 
            tmp[args.profile] = param[i]
            res = prf(tmp, 
                      fixed=args.profile,
                      nstw=int(nstw), 
                      nwalkers=nwalkers,
                      pool=pool,
                      cov=cov)
            xf[i,:] = np.insert(res, args.profile, param[i])
            tmp = np.array(xf[i,:],dtype='float64')
            chi2res[i] = chi2(xf[i,:])
            print(f"Partial ({i+1}/{numpts}): params={tmp}, and chi2={chi2res[i]}")
        
        # 6th: run from midpoint to left ---------------------------------------
        tmp = np.array(xf[numpts//2,:], dtype='float64')
        for i in range(numpts//2-1, -1, -1):
            tmp[args.profile] = param[i]
            res = prf(tmp, 
                      fixed=args.profile,
                      nstw=int(nstw), 
                      nwalkers=nwalkers,
                      pool=pool,
                      cov=cov)
            xf[i,:] = np.insert(res, args.profile, param[i])
            tmp = np.array(xf[i,:],dtype='float64')
            chi2res[i] = chi2(xf[i,:])
            print(f"Partial ({i+1}/{numpts}): params={tmp}, and chi2={chi2res[i]}")
        
        # 8th Append derived parameters ----------------------------------------
        xf = np.column_stack((xf, 
                              np.array([chi2v2(d) for d in xf], dtype='float64')))

        # 9th Save output file -------------------------------------------------    
        os.makedirs(os.path.dirname(f"{args.root}chains/"),exist_ok=True)
        hd = [names[args.profile],"chi2"] + names
        hd = hd + list(model.info()['likelihood'].keys()) + ["prior"]
        np.savetxt(f"{args.root}chains/{args.outroot}.{names[args.profile]}.txt",
                   np.concatenate([np.c_[param, chi2res],xf], axis=1),
                   fmt="%.9e",
                   header=f"nstw={args.nstw}, param={names[args.profile]}\n"+' '.join(hd),
                   comments="# ")
# ------------------------------------------------------------------------------
# ------------------------------------------------------------------------------
# ------------------------------------------------------------------------------
# ------------------------------------------------------------------------------
# ------------------------------------------------------------------------------
# ------------------------------------------------------------------------------