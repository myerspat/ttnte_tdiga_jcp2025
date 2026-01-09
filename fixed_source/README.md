# Fixed Source 

The fixed source problems presented in the journal paper are located in the following directories:

- `square/`: Homogeneous square with angular and mesh resolutions scaling studies.
- `circle/`: Homogeneous circle with angular and mesh resolutions scaling studies.
- `quarter_circle/`: Homogeneous quarter circle that is functionally the same as the homogeneous circle problem with angular and mesh resolution scaling studies.
- `cruciform/`: Shielded cruciform fixed source problem surrounded by void.

## Running the Scripts

These directories include the following scripts and directories:

- `square/`
  - `square.py`: Python script that runs the angular and mesh scaling studies using the `Runner` class from `runner.py` just under the `ttnte_tdiga_jcp2025` directory. This produces `direction/direction.jsonl`, `direction/meshes/`, `direction/rnorms/`, `meshsize/meshsize.jsonl`, `meshsize/meshes/`, and `meshsize/rnorms/`.
  - `postprocess.py`: Python script that computes errors using the angular flux solutions in `direction/meshes/` or `meshsize/meshes/` and adds them to the data in `direction/direction.jsonl` and `meshsize/meshsize.jsonl` and saves them in `direction/processed_direction.jsonl` and `meshsize/processed_meshsize.jsonl`.
  - `plot_direction.py`: Python script for angular resolution study plotting.
  - `direction/`: Directory for angular resolution results.
    - `direction.jsonl`: Ranks, compression, and timing results for all cases.
    - `processed_direction.jsonl`: Same as `direction.jsonl` with leakage fraction and all error metrics.
    - `meshes/`: Directory of angular flux solutions for each case. (Raw data used in the journal paper is stored in Zenodo)
    - `rnorms/`: Directory of residual history (after each restart only) for GMRES for each case. (Raw data used in the journal paper is stored in Zenodo)
    - `figs/`: Directory of figures.
  - `plot_meshsize.py`: Python script for mesh resolution study plotting.
  - `meshsize/`: Directory for mesh resolution results.
    - `meshsize.jsonl`: Ranks, compression, and timing results for all cases in the mesh resolution study.
    - `processed_meshsize.jsonl`: Same as `meshsize.jsonl` with leakage fraction and all error metrics.
    - `meshes/`: Directory of angular flux solutions for each case. (Raw data used in the journal paper is stored in Zenodo)
    - `rnorms/`: Directory of residual history (after each restart only) for GMRES for each case. (Raw data used in the journal paper is stored in Zenodo)
    - `figs/`: Directory of figures.
- `circle/`
  - `circle.py`: Python script that runs the angular and mesh scaling studies using the `Runner` class from `runner.py` just under the `ttnte_tdiga_jcp2025` directory. This produces `direction/direction.jsonl`, `direction/meshes/`, `direction/rnorms/`, `meshsize/meshsize.jsonl`, `meshsize/meshes/`, and `meshsize/rnorms/`.
  - `postprocess.py`: Python script that computes errors using the angular flux solutions in `direction/meshes/` or `meshsize/meshes/` and adds them to the data in `direction/direction.jsonl` and `meshsize/meshsize.jsonl` and saves them in `direction/processed_direction.jsonl` and `meshsize/processed_meshsize.jsonl`.
  - `plot_direction.py`: Python script for angular resolution study plotting.
  - `direction/`: Directory for angular resolution results.
    - `direction.jsonl`: Ranks, compression, and timing results for all cases.
    - `processed_direction.jsonl`: Same as `direction.jsonl` with leakage fraction and all error metrics.
    - `meshes/`: Directory of angular flux solutions for each case. (Raw data used in the journal paper is stored in Zenodo)
    - `rnorms/`: Directory of residual history (after each restart only) for GMRES for each case. (Raw data used in the journal paper is stored in Zenodo)
    - `figs/`: Directory of figures.
  - `plot_meshsize.py`: Python script for mesh resolution study plotting.
  - `meshsize/`: Directory for mesh resolution results.
    - `meshsize.jsonl`: Ranks, compression, and timing results for all cases in the mesh resolution study.
    - `processed_meshsize.jsonl`: Same as `meshsize.jsonl` with leakage fraction and all error metrics.
    - `meshes/`: Directory of angular flux solutions for each case. (Raw data used in the journal paper is stored in Zenodo)
    - `rnorms/`: Directory of residual history (after each restart only) for GMRES for each case. (Raw data used in the journal paper is stored in Zenodo)
    - `figs/`: Directory of figures.
- `quarter_circle/`
  - `quarter_circle.py`: Python script that runs the angular and mesh scaling studies using the `Runner` class from `runner.py` just under the `ttnte_tdiga_jcp2025` directory. This produces `direction/direction.jsonl`, `direction/meshes/`, `direction/rnorms/`, `meshsize/meshsize.jsonl`, `meshsize/meshes/`, and `meshsize/rnorms/`.
  - `postprocess.py`: Python script that computes errors using the angular flux solutions in `direction/meshes/` or `meshsize/meshes/` and adds them to the data in `direction/direction.jsonl` and `meshsize/meshsize.jsonl` and saves them in `direction/processed_direction.jsonl` and `meshsize/processed_meshsize.jsonl`.
  - `plot_direction.py`: Python script for angular resolution study plotting.
  - `direction/`: Directory for angular resolution results.
    - `direction.jsonl`: Ranks, compression, and timing results for all cases.
    - `processed_direction.jsonl`: Same as `direction.jsonl` with leakage fraction and all error metrics.
    - `meshes/`: Directory of angular flux solutions for each case. (Raw data used in the journal paper is stored in Zenodo)
    - `rnorms/`: Directory of residual history (after each restart only) for GMRES for each case. (Raw data used in the journal paper is stored in Zenodo)
    - `figs/`: Directory of figures.
  - `plot_meshsize.py`: Python script for mesh resolution study plotting.
  - `meshsize/`: Directory for mesh resolution results.
    - `meshsize.jsonl`: Ranks, compression, and timing results for all cases in the mesh resolution study.
    - `processed_meshsize.jsonl`: Same as `meshsize.jsonl` with leakage fraction and all error metrics.
    - `meshes/`: Directory of angular flux solutions for each case. (Raw data used in the journal paper is stored in Zenodo)
    - `rnorms/`: Directory of residual history (after each restart only) for GMRES for each case. (Raw data used in the journal paper is stored in Zenodo)
    - `figs/`: Directory of figures.
- `cruciform/`
  - `cruciform.py`: Python script that runs the cases presented in the journal paper for the shielded cruciform fixed source problem. This script saves the angular flux solution, TT-ranks, compression, and timing results and saves them to `data.pkl`.
  - `plot.py`: Python script that computes the leakage fraction and error metrics and creates any plots. All metrics are saved to `data.pkl`.
  - `data.pkl`: All solutions, errors, TT-ranks, compression, and timing results. (Raw data used in the journal paper is stored in Zenodo)
  - `figs/`: Directory for figures.

   
We note the solutions for each case in `meshes/` and the residuals in `rnorms/` are labeled `N{num_ordinates}_G{num_groups}_A{knot_spans_xhat + degree_xhat}_B{knot_spans_yhat + degree_yhat}_p{degree_xhat}_q{degree_yhat}_eps{eps}{device}.pkl` where `num_ordinates` is the number of ordinates, `num_groups` is the number of energy groups, `knot_spans_xhat` is the number of knot spans along the parametric x-axis, `knot_spans_yhat` is the number of knot spans along the parametric y-axis, `degree_xhat` is the NURBS basis function degree along the parametric x-axis, `degree_yhat` is the NURBS basis function degree along the parametric y-axis, `eps` is the truncation tolerance, `device` is either `cpu` or `gpu` depending on what device we solved that instance. Each pickle has each case (CSR, TT, Mixed, TT (rounded), or Mixed (rounded)) as a dictionary key to a Numpy array. Only `eps = 1e-8` have solutions computed with CSR. The plotting scripts (`plot_direction.py` and `plot_meshsize.py`) require a path to their OpenMC reference solution and standard deviation (if applicable). They are currently configured to find their respective OpenMC solutions using a relative path if `ttnte` shares a directory with `ttnte_tdiga_jcp2025`.
