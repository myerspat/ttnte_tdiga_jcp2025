import multiprocessing
import sys

if __name__ == "__main__":
    multiprocessing.set_start_method("spawn")
    sys.path.append("../..")

import numpy as np
from quarter_circle import get_mesh, get_xs

from process import process

if __name__ == "__main__":
    # Get MC solutions
    leakage_frac_openmc = [0.43995486599999994, 2.2442114486922458e-05]
    phi_mc = np.load(
        "../../../ttnte/notebooks/fixed_source/quarter_circle/openmc/data/mesh_flux.npy"
    )
    phi_mc_stdev = np.load(
        "../../../ttnte/notebooks/fixed_source/quarter_circle/openmc/data/mesh_stdev.npy"
    )

    # Process results
    process(
        get_xs, get_mesh, "./direction", leakage_frac_openmc, [phi_mc, phi_mc_stdev]
    )

    # Process results
    process(get_xs, get_mesh, "./meshsize", leakage_frac_openmc, [phi_mc, phi_mc_stdev])
