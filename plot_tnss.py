import os
import json
import pickle
import copy
import time
from itertools import product
from pathlib import Path
from typing import List, Dict

import numpy as np
import matplotlib.pyplot as plt


def pretty_eps(eps):
    for i in range(12):
        if eps == 10**-i:
            return "10^{-" + str(i) + "}"


if __name__ == "__main__":
    # Truncation tolerances to consider
    epss = [float(f"1e-{i}") for i in range(8, 0, -1)]
    degrees = [2, 3, 4, 6]

    # Plotting markers and line styles
    markers = ["o", "s", "*"]
    linestyles = ["-", "--", "-."]
    lines = [linestyle + marker for linestyle, marker in product(linestyles, markers)]

    # =======================================================
    # Angular and mesh resolution studies
    # =======================================================
    # Directories to search
    dirs = [
        Path(dir)
        for dir in [
            "fixed_source/square",
            "fixed_source/circle",
            "fixed_source/quarter_circle",
        ]
    ]

    for dir in dirs:
        for subdir in ["direction/meshes", "meshsize/meshes"]:
            print("=" * 80 + "\nDirectory: {}".format(dir / subdir))

            # Iterate through all the degrees
            for degree in degrees:
                data = {
                    "x": [],
                    "cr": [],
                    "time": [],
                    "error": [],
                }

                with open(
                    dir / subdir.split("/")[0] / "tnss_data.jsonl",
                    "r",
                    encoding="utf-8",
                ) as f:
                    for line in f:
                        # Read line
                        line_data = json.loads(line)

                        # Check if this is the data we want
                        if line_data["degree"] != degree:
                            continue

                        # Add x-axis data
                        data["x"].append(
                            line_data["num_ordinates"]
                            if subdir == "direction/meshes"
                            else (line_data["factor"] + degree) ** 2
                        )

                        # Add results to the correct part of the dictionary
                        for key in ["cr", "time", "error"]:
                            data[key].append(line_data[key])

                # Handle additional patches in quarter circle
                if dir.stem == "quarter_circle":
                    data["x"] = [4 * x for x in data["x"]]

                # Plot compression ratio
                plt.clf()
                for i in range(len(epss)):
                    plt.plot(
                        np.array(data["x"]),
                        np.array(data["cr"])[:, i],
                        np.array(lines)[i],
                        label=rf"$\epsilon = {pretty_eps(epss[i])}$",
                    )
                plt.xlabel(
                    r"$N_\Omega$"
                    if subdir == "direction/meshes"
                    else r"$N_e \times N_{\hat x} \times N_{\hat y}$"
                )
                plt.ylabel("Compression Ratio")
                plt.xscale("log")
                plt.yscale("log")
                plt.legend(ncol=2)
                plt.savefig(
                    dir / subdir.split("/")[0] / f"figs/tnss_cr_{degree}.png", dpi=300
                )

                # Plot TN-SS times
                plt.clf()
                for i in range(len(epss)):
                    plt.plot(
                        np.array(data["x"]),
                        np.array(data["time"])[:, i],
                        np.array(lines)[i],
                        label=rf"$\epsilon = {pretty_eps(epss[i])}$",
                    )
                plt.xlabel(
                    r"$N_\Omega$"
                    if subdir == "direction/meshes"
                    else r"$N_e \times N_{\hat x} \times N_{\hat y}$"
                )
                plt.ylabel("TN-SS Time")
                plt.xscale("log")
                plt.yscale("log")
                plt.legend(ncol=2)
                plt.savefig(
                    dir / subdir.split("/")[0] / f"figs/tnss_time_{degree}.png", dpi=300
                )

                # Plot TN-SS error
                plt.clf()
                for i in range(len(epss)):
                    plt.plot(
                        np.array(data["x"]),
                        np.array(data["error"])[:, i],
                        np.array(lines)[i],
                        label=rf"$\epsilon = {pretty_eps(epss[i])}$",
                    )
                plt.xlabel(
                    r"$N_\Omega$"
                    if subdir == "direction/meshes"
                    else r"$N_e \times N_{\hat x} \times N_{\hat y}$"
                )
                plt.ylabel("TN-SS Error")
                plt.xscale("log")
                plt.yscale("log")
                plt.legend(ncol=2)
                plt.savefig(
                    dir / subdir.split("/")[0] / f"figs/tnss_error_{degree}.png",
                    dpi=300,
                )
