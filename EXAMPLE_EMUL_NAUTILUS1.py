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
import argparse, random
import numpy as np
from cobaya.yaml import yaml_load
from cobaya.model import get_model
from nautilus import Prior, Sampler
from getdist import loadMCSamples
# ------------------------------------------------------------------------------
# ------------------------------------------------------------------------------
# ------------------------------------------------------------------------------
# ------------------------------------------------------------------------------
# ------------------------------------------------------------------------------
# ------------------------------------------------------------------------------
parser = argparse.ArgumentParser(prog='EXAMPLE_PROJECT_NAUTILUS1')
parser.add_argument("--root",
                    dest="root",
                    help="Name of the Output File",
                    nargs='?',
                    const=1,
                    default="./projects/lsst_y1/")
parser.add_argument("--outroot",
                    dest="outroot",
                    help="Name of the Output File",
                    nargs='?',
                    const=1,
                    default="example_nautilus1")
parser.add_argument("--nlive",
                    dest="nlive",
                    help="Number of live points ",
                    type=int,
                    nargs='?',
                    const=1,
                    default=1000)
parser.add_argument("--maxfeval",
                    dest="maxfeval",
                    help="Minimizer: maximum number of likelihood evaluations",
                    type=int,
                    nargs='?',
                    const=1,
                    default=100000)
parser.add_argument("--neff",
                    dest="neff",
                    help="Minimum effective sample size. ",
                    type=int,
                    nargs='?',
                    const=1,
                    default=10000)
parser.add_argument("--flive",
                    dest="flive",
                    help="Maximum fraction of the evidence contained in the live set before building the initial shells terminates",
                    type=float,
                    nargs='?',
                    const=1,
                    default=0.01)
parser.add_argument("--nnetworks",
                    dest="nnetworks",
                    help="Number of Neural Networks",
                    type=int,
                    nargs='?',
                    const=1,
                    default=4)
args, unknown = parser.parse_known_args()
# ------------------------------------------------------------------------------
# ------------------------------------------------------------------------------
# ------------------------------------------------------------------------------
# ------------------------------------------------------------------------------
# ------------------------------------------------------------------------------
# ------------------------------------------------------------------------------
yaml_string=r"""
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
      return 1.e20
    res2 = model.loglike(point,
                         make_finite=False,
                         cached=False,
                         return_derived=False)
    if np.isinf(res2) or  np.any(np.isnan(res2)):
      return 1e20
    return -2.0*(res1+res2)

def likelihood(params):
  res = chi2(params)
  if (res > 1.e19 or np.isinf(res) or  np.isnan(res)):
    return -np.inf
  else:
    return -0.5*res
# ------------------------------------------------------------------------------
# ------------------------------------------------------------------------------
# ------------------------------------------------------------------------------
# ------------------------------------------------------------------------------
# ------------------------------------------------------------------------------
# ------------------------------------------------------------------------------
from mpi4py.futures import MPIPoolExecutor

if __name__ == '__main__':
    print(f"nlive={args.nlive}, output={args.root}chains/{args.outroot}")
    # Build Nautilus Prior from Cobaya
    NautilusPrior = Prior()                                       # Nautilus Call 
    dim    = model.prior.d()                                      # Cobaya call
    bounds = model.prior.bounds(confidence=0.999999)              # Cobaya call
    names  = list(model.parameterization.sampled_params().keys()) # Cobaya Call
    print(names)
    for b, name in zip(bounds, names):
      NautilusPrior.add_parameter(name, dist=(b[0], b[1]))
    
    sampler = Sampler(NautilusPrior, 
                      likelihood,  
                      filepath=f"{args.root}chains/{args.outroot}_checkpoint.hdf5", 
                      n_dim=dim,
                      pool=MPIPoolExecutor(),
                      n_live=args.nlive,
                      n_networks=args.nnetworks,
                      resume=True)
    sampler.run(f_live=args.flive,
                n_eff=args.neff,
                n_like_max=args.maxfeval,
                verbose=True,
                discard_exploration=True)
    points, log_w, log_l = sampler.posterior()
    
    # Save output file ---------------------------------------------------------
    os.makedirs(os.path.dirname(f"{args.root}chains/"),exist_ok=True)
    np.savetxt(f"{args.root}chains/{args.outroot}.1.txt",
               np.column_stack((np.exp(log_w), log_l, points, -2*log_l)),
               fmt="%.5e",
               header=f"nlive={args.nlive}, maxfeval={args.maxfeval}, log-Z ={sampler.log_z}\n"+' '.join(names),
               comments="# ")
    
    # Save a range files -------------------------------------------------------
    rows = [(str(n),float(l),float(h)) for n,l,h in zip(names,bounds[:,0],bounds[:,1])]
    with open(f"{args.root}chains/{args.outroot}.ranges", "w") as f: 
      f.writelines(f"{n} {l:.5e} {h:.5e}\n" for n, l, h in rows)

    # Save a paramname files ---------------------------------------------------
    param_info = model.info()['params']
    latex  = [param_info[x]['latex'] for x in names]
    names.append("chi2*")
    latex.append("\\chi^2")
    np.savetxt(f"{args.root}chains/{args.outroot}.paramnames", 
               np.column_stack((names,latex)),
               fmt="%s")

    # Save a cov matrix --------------------------------------------------------
    samples = loadMCSamples(f"{args.root}chains/{args.outroot}",
                            settings={'ignore_rows': u'0.0'})
    np.savetxt(f"{args.root}chains/{args.outroot}.covmat",
               np.array(samples.cov(), dtype='float64'),
               fmt="%.5e",
               header=' '.join(names),
               comments="# ")
# ------------------------------------------------------------------------------
# ------------------------------------------------------------------------------
# ------------------------------------------------------------------------------
# ------------------------------------------------------------------------------
# ------------------------------------------------------------------------------
# ------------------------------------------------------------------------------