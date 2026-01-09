# `ttnte_tdiga_jcp2025`

This repository stores all figures and scripts to generate the results for the submission of "Tensorized Discontinuous Isogeometric Analysis Method for the 2-D Time-Independent Linearized Boltzmann Transport Equation".
- Authors: Patrick A. Myers, Joseph A. Bogdan, Majdi I. Redaideh, and Brian C. Kiedrowski
- Corresponding Authors: Patrick A. Myers (myerspat@umich.edu) and Brian C. Kiedrowski (bckiedro@umich.edu)
- Target Journal: Journal of Computational Physics

This repository also has the results for our 2026 PHSYOR submission "Tensor Train Decomposition Applied to the Isogeometric Discontinuous Galerkin 2-D Neutron Transport Equation". All results for that are located in `eigenvalue/`.
- Authors: Patrick A. Myers, Majdi I. Redaideh, and Brian C. Kiedrowski
- Corresponding Authors: Patrick A. Myers (myerspat@umich.edu) and Brian C. Kiedrowski (bckiedro@umich.edu)

Specifications of the cluster used to generate these results:
- CPU: AMD Ryzen Threadripper PRO 7985WX 64-Cores
- GPU: NVIDIA RTX PRO 6000 Blackwell Max-Q
- RAM: (8) Micron 64GB DDR5 5600 MT/s (Configured memory speed of 5200 MT/s)
  - Totaling 512 GB of RAM

## Installation and Setup
We recommend creating a new Python environment for this code. Clone commit `a19363b` for [`ttnte`](https://github.com/myerspat/ttnte/tree/a19363b610a760d1d8dd587c310cdca9739968ad):
```shell
git clone https://github.com/myerspat/ttnte.git
cd ttnte && git checkout a19363b
```

Install CUDA capable Pytorch (both Python with pip and libtorch) that matches you GPU's drivers by following [their documentation](https://pytorch.org/get-started/locally/). Then install [`igakit`](https://github.com/dalcinl/igakit), [`geomdl`](https://github.com/orbingol/NURBS-Python), [`torchtt`](https://github.com/ion-g-ion/torchTT):
```shell
pip install https://github.com/dalcinl/igakit/archive/refs/heads/master.zip
SETUPTOOLS_USE_CYTHON=1 pip install git+https://github.com/orbingol/NURBS-Python.git
pip install git+https://github.com/ion-g-ion/torchTT
```

Ensure `geomdl` compiled with Cython and the C++ backend off `torchtt` compiled by running the following:
```python
try:
    from geomdl.core import NURBS
    print("geomdl successfully compiled with Cython")
except:
    print("geomdl failed to compiled with Cython")

import torchtt
if torchtt.cpp_enabled():
    print("C++ backend of torchtt successfully compiled")
else:
    print("C++ backend of torchtt failed to compiled")
```

For the full environment used to generate the results please see `environment.yml`. Ensure your environment is aware of your CUDA drivers by running `which nvcc`. You should see something of the sort `/path/to/nvcc`. Fill in the path to libtorch for the `TORCH_INSTALL_PREFIX` environment variable and run the following to install `ttnte`:
```shell
export TORCH_INSTALL_PREFIX=/path/to/libtorch
export _GLIBCXX_USE_CXX11_ABI=$(python3 -c "import torch; print(int(torch._C._GLIBCXX_USE_CXX11_ABI))")
TTNTE_CPP_BACKEND=1 pip install ".[dev]" -v
```

`ttnte` is now installed and its C++ backend is compiled. Now we can clone `ttnte_tdiga_jcp2025`:
```shell
cd ..
git clone https://github.com/myerspat/ttnte_tdiga_jcp2025.git
cd ttnte_tdiga_jcp2025
```

## Script and Results File Structure

### Primary Scripts

- `environment.yml`: Conda environment used in creating the results presented in the paper.
- `runner.py`: Main Python script for angular and mesh resolution scaling studies. Never called directly but imported into other scripts.
- `process.py`: Python script for taking the raw data of the angular and mesh resolution studies and computing leakage fractions, errors, and other post processing data. Never called directly but imported into other scripts.
- `extract.py`: Python script with extraction methods for angular and mesh resolution data. Used for plotting. Never called directly but imported into other scripts.
- `run_scripts.sh`: Generate all results. We note that even on our machine some of the problems were too big to fit into RAM or onto the GPU. There are try/except blocks that catch the majority of the problems but this can still crash mid run. `runner.py` should save each run once their completed so no prior runs are lost when this occurs except for the case the `Runner` is currently on. Best way to handle this is to remove that case and continue.

### Directories

- `fixed_source/`: Contains all scripts and figures for the fixed source problems. Refer to that directory for specifics.
- `eigenvalue/`: Contains all scripts and figures for the eigenvalue problems. Refer to that directory for specifics.
- `other/`: Supporting scripts and figures used in the Background of the paper.

## Raw Data

Add description of this once it is on Zenodo.
