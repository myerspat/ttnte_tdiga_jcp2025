# k-Eigenvalue Numerical Examples

The $k$-eigenvalue problems presented in the journal paper and the conference paper are located in the following directories:

- `circle/`: The homogeneous circle problem from Section 4.1.1 of ["Analytical benchmark test set for criticality code verification"](https://doi-org.proxy.lib.umich.edu/10.1016/S0149-1970(02)00098-7).
- `quarter_circle/`: The same problem represented as a quarter circle with reflecting boundary conditions.
- `pincell/`: Infinite array of [C5G7](https://www.oecd-nea.org/jcms/pl_13548/benchmark-on-deterministic-transport-calculations-without-spatial-homogenisation?details=true) fuel.
- `lightbridge_ba/`: Infinite array of cruciform fuel based on the [four-lobe Lightbridge design](https://www-tandfonline-com.proxy.lib.umich.edu/doi/abs/10.13182/NT12-A15354) with a burnable absorber (BA) displacer. The cross sections are taken from the [7-group KAIST 2B benchmark](https://github.com/nzcho/Nurapt-Archives/blob/master/KAIST-Benchmark-Problems/README.md).
- `lightbridge_gas/`: Infinite array of cruciform fuel based on the [four-lobe Lightbridge design](https://www-tandfonline-com.proxy.lib.umich.edu/doi/abs/10.13182/NT12-A15354) with a gas displacer. The cross sections are taken from the [7-group KAIST 2B benchmark](https://github.com/nzcho/Nurapt-Archives/blob/master/KAIST-Benchmark-Problems/README.md).

## Running the Scripts

We note these examples are divided into conference and journal paper as to when they were written and ran. For the PHYSOR conference paper we have the following scripts within each directory:

- `circle/`
  - `circle.py`: Script that runs the cases presented in the PHYSOR paper, computes errors, and creates figures.
  - `stats.pkl`: Error and timing results. (Raw data used in PHYSOR paper is stored in Zenodo)
  - `solutions.pkl`: Angular flux solutions for each case. (Raw data used in PHYSOR paper is stored in Zenodo)
  - `figs/`: Directory for figures.
- `lightbridge_gas/`
  - `lightbridge_gas.py`: Script that runs the cases presented in the PHYSOR and journal papers, computes errors, and creates figures.
  - `stats.pkl`: Error and timing results. (Raw data used in PHYSOR paper is stored in Zenodo)
  - `solutions.pkl`: Angular flux solutions for each case. (Raw data used in PHYSOR paper is stored in Zenodo)
  - `figs/`: Directory for figures.

The results in `lightbridge_gas/` are shown in both the PHYSOR and journal papers with the journal paper going in more detail. To generate the results simply run `python circle.py` or `python lightbridge_gas.py`. We note `lightbridge_gas.py` requires paths to the OpenMC reference solution and standard deviation. Assuming `ttnte` and `ttnte_tdiga_jcp2025` are in the same directory the current scripts will find the OpenMC reference solution stored in `ttnte` using a relative path. The other results presented in the journal paper include the following directories and scripts:

- `quarter_circle/`
  - `quarter_circle.py`: Python script for generating the solution for each case presented in the journal paper for the quarter circle problem, as well as producing timing results.
  - `plot.py`: Python script for computing errors and plotting.
  - `data.pkl`: All solutions, errors, and timing results.
  - `figs/`: Directory for figures.
- `pincell/`
  - `pincell.py`: Python script for generating the solution for each case presented in the journal paper for the pincell problem, as well as producing timing results.
  - `plot.py`: Python script for computing errors and plotting.
  - `data.pkl`: All solutions, errors, and timing results.
  - `figs/`: Directory for figures.
- `lightbridge_ba/`
  - `lightbridge_ba.py`: Python script for generating the solution for each case presented in the journal paper for the infinite array of cruciform fuel with a BA displacer problem, as well as producing timing results.
  - `plot.py`: Python script for computing errors and plotting.
  - `data.pkl`: All solutions, errors, and timing results.
  - `figs/`: Directory for figures.
