import sys
import multiprocessing

if __name__ == "__main__":
    multiprocessing.set_start_method("forkserver")
    sys.path.append("../..")

from circle import get_xs, get_mesh
from process import process

if __name__ == "__main__":
    # Get MC solutions
    leakage_frac_openmc = [0.43995423399999983, 2.2245143201699137e-05]

    # Process results
    process(get_xs, get_mesh, "./direction", leakage_frac_openmc)

    # Process results
    process(get_xs, get_mesh, "./meshsize", leakage_frac_openmc)
