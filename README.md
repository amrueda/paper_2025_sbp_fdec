* Make sure to run the following tests with Julia 1.11.7.

* Clone the reproducibility repository and navigate to the directory:
```bash
git clone git@github.com:amrueda/paper_2025_sbp_fdec.git
cd paper_2025_sbp_fdec
```
* Instantiate the code:
```bash
julia --project=./code -e 'import Pkg; Pkg.instantiate()'
```
* The following convergenge tests output XML files with three columns. The first column is the data for $E_x$, the second one for $E_y$ and the third one for $B_z$.

* Run the convergence tests for the periodic case in the weak form variant with SSPRK time integration for one element:
```bash
julia --project=./code -e 'include(joinpath("code", "examples", "convergence_weak_form.jl"))'
```

* Run the convergence tests for the non-periodic case in the weak form variant with SSPRK time integration for one element:
```bash
julia --project=./code -e 'include(joinpath("code", "examples", "convergence_weak_form_np.jl"))'
```

* Run the convergence tests for the periodic case in the weak form variant with Crank-Nicolson time integration for one element:
```bash
julia --project=./code -e 'include(joinpath("code", "examples", "convergence_weak_form_implicit.jl"))'
```

* Run the convergence tests for the periodic case in the strong form variant with SSPRK time integration for one element as validation of the equivalence to the weak form. This test is not shown in the paper since the results are the same up to machine precision.
```bash
julia --project=./code -e 'include(joinpath("code", "examples", "convergence_strong_form.jl"))'
```

* Run the convergence tests for the periodic case in the weak form variant with SSPRK time integration for a constant number of nodes per element. This runs several convergence tests for varying amounts of points:
```bash
julia --project=./code -e 'include(joinpath("code", "examples", "convergence_var_nodes.jl"))'
```

* Run the convergence tests for the non-periodic case in the weak form variant with SSPRK time integration for a constant number of nodes per element. This runs several convergence tests for varying amounts of points:
```bash
julia --project=./code -e 'include(joinpath("code", "examples", "convergence_var_nodes_np.jl"))'
```

* The last two tests output the divergence and energy plots:

* Run the energy and divergence tests for the periodic case to T = 1 for a coarse and a fine resolution:
```bash
julia --project=./code -e 'include(joinpath("code", "examples", "div_energy_coarse_fine.jl"))'
```

* Run the energy and divergence tests for the non-periodic case to T = 1 for a fine resolution:
```bash
julia --project=./code -e 'include(joinpath("code", "examples", "div_energy_fine_np.jl"))'
```

* Run the energy and divergence tests to T = 10000 for a coarse resolution:
```bash
julia --project=./code -e 'include(joinpath("code", "examples", "div_energy_test_T10000.jl"))'
```

* Run the energy tests for the non-SBP operator to T = 1 for a coarse resolution:
```bash
julia --project=./code -e 'include(joinpath("code", "examples", "energy_coarse_non_sbp.jl"))'
```

* Run a detailed convergence test for the non-periodic case for the magnetic field:
```bash
julia --project=./code -e 'include(joinpath("code", "examples", "convergence_study_B_field.jl"))'
```