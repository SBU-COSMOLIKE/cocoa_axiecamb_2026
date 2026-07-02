## Running projects (Basic instructions) <a name="axicamb_running_cosmolike_projects"></a> 

From `Cocoa/Readme` instructions:

> [!Note]
> To activate AxieCAMB projects, comment the following lines on `set_installation_options.sh` (they are not active by default) before running `setup_cocoa.sh` and `compile_cocoa.sh`
> 
>     [Adapted from Cocoa/set_installation_options.sh shell script]
>     (...)
>
>     # ------------------------------------------------------------------------------
>     # The keys below control which private projects (may not be public repo)
>     # ------------------------------------------------------------------------------
>     (...)
>     #export INSTALL_AXIE_CAMB_V2=1
>     #export INSTALL_AXIE_CAMB_2026_PROJECT=1
>

> [!Note]
>  Adjust the following lines on `set_installation_options.sh` if you are using non-standard repositories
> 
>     [Adapted from Cocoa/set_installation_options.sh shell script]
>     (...)
>
>     # ------------------------------------------------------------------------------
>     # URL of private projects below ------------------------------------------------
>     # ------------------------------------------------------------------------------
>     export AXIONS_2025_PROJECT_URL="https://github.com/SBU-COSMOLIKE/cocoa_axions.git"
>     export AXIONS_PROJECT_NAME="axions"
>
>     export AXIE_CAMB_2026_PROJECT_URL="https://github.com/SBU-COSMOLIKE/cocoa_axiecamb_2026.git"
>     export AXIE_CAMB_2026_PROJECT_NAME="axicambv2"
>
>     (...)
>
>     # ------------------------------------------------------------------------------
>     # GENERAL PACKAGE URL AND VERSIONS. CHANGES IN THE COMMIT ID MAY BREAK COCOA ---
>     # ------------------------------------------------------------------------------
>     (...)
>     export AXIE_CAMB_URL="https://github.com/SBU-COSMOLIKE/NewTestingAxieCAMB.git"
>     export AXIE_CAMB_GIT_COMMIT="5007a6d228f36bc68cf488b41798764f169de632"
>     export AXIE_CAMB_NAME="axiecamb"
>
>     export AXION_HMCODE_URL="https://github.com/SBU-COSMOLIKE/axionHMcode.git"
>     export AXION_HMCODE_GIT_COMMIT="a85ba2679cba5abe68b4cfd1e49b7a52bdcda424"
>     export AXION_HMCODE_NAME="axionHMcode"


To run the example

 **Step :one:**: activate the cocoa Conda environment,  and the private Python environment 

      conda activate cocoa

and

      source start_cocoa.sh
 
 **Step :two:**: Select the number of OpenMP cores (below, we set it to 8).

    export OMP_PROC_BIND=close; export OMP_NUM_THREADS=8; export OMP_PLACES=cores; export OMP_DYNAMIC=FALSE
 
 **Step :three:**: The folder `projects/axicambv2` contains examples. So, run the `cobaya-run` on the first example following the commands below.

- **One model evaluation**:

  - Linux

        mpirun -n 1 --oversubscribe --mca pml ^ucx --mca btl vader,tcp,self --report-bindings \
           --bind-to core:overload-allowed --rank-by slot --map-by numa:pe=${OMP_NUM_THREADS} \
           cobaya-run ./projects/axicambv2/EXAMPLE_EVALUATE1.yaml -f

  -  macOS (arm)

         mpirun -n 1 --oversubscribe cobaya-run ./projects/axicambv2/EXAMPLE_EVALUATE1.yaml -f

- **MCMC (Metropolis-Hastings Algorithm)**:

  - Linux

        mpirun -n 4 --oversubscribe --mca pml ^ucx --mca btl vader,tcp,self --report-bindings \
           --bind-to core:overload-allowed --rank-by slot --map-by numa:pe=${OMP_NUM_THREADS} \
           cobaya-run ./projects/axicambv2/EXAMPLE_MCMC1.yaml -f

   -  macOS (arm)
     
          mpirun -n 4 --oversubscribe cobaya-run ./projects/axicambv2/EXAMPLE_MCMC1.yaml -f

- **Nautilus**:

  - Linux
    
        mpirun -n 90 --oversubscribe --mca pml ^ucx --mca btl vader,tcp,self \
            --bind-to core:overload-allowed --rank-by slot --map-by slot:pe=${OMP_NUM_THREADS} \
            python -m mpi4py.futures ./projects/axicambv2/EXAMPLE_NAUTILUS1.py \
                --root ./projects/axicambv2/ --outroot "EXAMPLE_NAUTILUS1"  \
                --maxfeval 750000 --nlive 2048 --neff 15000 --flive 0.01 --nnetworks 5

  - macOS (arm)

        mpirun -n 12 --oversubscribe python -m mpi4py.futures ./projects/axicambv2/EXAMPLE_NAUTILUS1.py \
                --root ./projects/axicambv2/ --outroot "EXAMPLE_NAUTILUS1"  \
                --maxfeval 750000 --nlive 2048 --neff 15000 --flive 0.01 --nnetworks 5

- **Global Minimizer**:

  Our minimizer is a reimplementation of `Procoli`, developed by Karwal et al (arXiv:2401.14225) 

  - Linux
    
        mpirun -n 51 --oversubscribe --mca pml ^ucx --mca btl vader,tcp,self \
            --bind-to core:overload-allowed --rank-by slot --map-by slot:pe=${OMP_NUM_THREADS} \
            python ./projects/axicambv2/EXAMPLE_MINIMIZE1.py --root ./projects/axicambv2/ \
                --outroot "EXAMPLE_MIN1" --nstw 450

  - macOS (arm)

        mpirun -n 12 python ./projects/axicambv2/EXAMPLE_MINIMIZE1.py --root ./projects/axicambv2/ \
              --outroot "EXAMPLE_MIN1" --nstw 450

  The number of steps per Emcee walker per temperature is $n_{\\rm stw}$,
  and the number of walkers is $n_{\\rm w}={\\rm max}(3n_{\\rm params},n_{\\rm MPI})$.
  The minimum number of total evaluations is $3n_{\\rm params} \times n_{\rm T} \times n_{\\rm stw}$, which can be distributed among $n_{\\rm MPI} = 3n_{\\rm params}$ MPI processes for faster results.
    

- **Profile**: 

  - Linux
    
          mpirun -n 51 --oversubscribe --mca pml ^ucx --mca btl vader,tcp,self \
            --bind-to core:overload-allowed --rank-by slot --map-by slot:pe=${OMP_NUM_THREADS} \
            python ./projects/axicambv2/EXAMPLE_PROFILE1.py \
              --root ./projects/axicambv2/ --cov 'chains/EXAMPLE_MCMC1.covmat' \
              --outroot "EXAMPLE_PROFILE1" --factor 3 --nstw 450 --numpts 10 \
              --profile ${SLURM_ARRAY_TASK_ID} \
              --minfile="./projects/axicambv2/chains/EXAMPLE_MIN1.txt"

  -  macOS (arm)

          mpirun -n 51 --oversubscribe python ./projects/axicambv2/EXAMPLE_PROFILE1.py \
              --root ./projects/axicambv2/ --cov 'chains/EXAMPLE_MCMC1.covmat' \
              --outroot "EXAMPLE_PROFILE1" --factor 3 --nstw 450 --numpts 10 \
              --profile ${SLURM_ARRAY_TASK_ID} \
              --minfile="./projects/axicambv2/chains/EXAMPLE_MIN1.txt"

  The argument `factor` specifies the start and end of the parameter being profiled:

      start value ~ mininum value - factor*np.sqrt(np.diag(cov))
      end   value ~ mininum value + factor*np.sqrt(np.diag(cov))

  We advise ${\rm factor} \sim 3$ for parameters that are well constrained by the data when a covariance matrix is provided.
  If `cov` is not supplied, the code estimates one internally from the prior.
  If a parameter is poorly constrained or `cov` is not given, we recommend ${\rm factor} \ll 1$.

> [!Warning]
> When running Profiles, you should not set flat priors on parameters that are not well constrained by the data. 
> By doing that, you then risk having the minimizer select values near the boundary of parameter space. 
