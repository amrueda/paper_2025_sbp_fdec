* Clone the reproducibility repository:
```bash
git clone git@github.com:amrueda/paper_2025_sbp_fdec.git
```
* Instantiate the code:
```bash
julia --project=./code -e 'import Pkg; Pkg.instantiate()'
```
* The following convergenge tests output XML files with three columns. The first column is the data for $E_x$, the second one for $E_y$ and the third one for $B_z$.

* Run the convergence tests for the weak form variant with SSPRK time integration for one element:
```bash
julia --project=./code -e 'include(joinpath("code", "examples", "convergence_weak_form.jl"))'
```

* Run the convergence tests for the weak form variant with Crank-Nicolson time integration for one element:
```bash
julia --project=./code -e 'include(joinpath("code", "examples", "convergence_weak_form_implicit.jl"))'
```

* Run the convergence tests for the strong form variant with Crank-Nicolson time integration for one element as validation of the equivalence to the weak form. This test is not shown in the paper since the results are the same up to machine precision.
```bash
julia --project=./code -e 'include(joinpath("code", "examples", "convergence_strong_form.jl"))'
```

* Run the convergence tests for the weak form variant with SSPRK time integration for a constant number of nodes per element. This runs several convergence tests for varying amount of points:
```bash
julia --project=./code -e 'include(joinpath("code", "examples", "convergence_var_nodes.jl"))'
```

* Run the energy and divergence tests to T = 1 for a coarse and a fine resolution:
```bash
julia --project=./code -e 'include(joinpath("code", "examples", "div_energy_coarse_fine.jl"))'
```

* Run the energy and divergence tests to T = 10000 for a coarse resolution:
```bash
julia --project=./code -e 'include(joinpath("code", "examples", "div_energy_test_T10000.jl"))'
```