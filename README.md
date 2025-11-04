* Clone the reproducibility repository:
```bash
git clone git@github.com:amrueda/paper_2025_sbp_fdec.git
```
* Instantiate the code and run a test:
```bash
julia --project=./code -e 'import Pkg; Pkg.instantiate()'
julia --project=./code -e 'include(joinpath("code", "examples", "convergence_sparse.jl"))'
```
