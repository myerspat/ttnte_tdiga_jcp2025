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

from pytens import Index, Tensor, TreeNetwork
from pytens.search.configuration import SearchConfig
from pytens.search.search import SearchEngine


def tnss(
    config: SearchConfig,
    epss: List[float],
    tn_diagrams_dir: Path,
    tn_actions_dir: Path,
    file_path: Path,
    discretization: Dict,
):
    """"""
    # Load pickled data
    with open(file_path, "rb") as f:
        data = pickle.load(f)

    if "CSR" not in data and (
        "solve_method" in data and "CSR" not in data["solve_method"]
    ):
        return None

    dis = list(discretization.values())
    print(
        "-" * 80
        + "\nN = {}, G = {}, P = {}, A = {}, B = {}".format(
            int(dis[0] * dis[1] * dis[2]), dis[3], dis[4], dis[5], dis[6]
        )
    )

    # Get CSR data
    psi = (
        (data["CSR"] if not isinstance(data["CSR"], tuple) else data["CSR"][0])
        if "solve_method" not in data
        else data["psi"]["value"][
            np.argwhere(np.array(data["solve_method"]) == "CSR").flatten()[0]
        ]
    ).reshape([size for size in discretization.values() if size != 1])
    psi = Tensor(
        psi if isinstance(psi, np.ndarray) else psi.numpy(),
        [
            Index(dimension, size)
            for dimension, size in discretization.items()
            if size != 1
        ],
    )

    # Run TN-SS partition for all truncation tolerances
    data = {"cr": [], "time": [], "error": []}
    for config.engine.eps in epss:
        print(f"-- eps = {config.engine.eps},", end=" ")

        # Create initial TN
        net = TreeNetwork()
        net.add_node("T0", copy.deepcopy(psi))

        start = time.time()
        engine = SearchEngine(config)
        result = engine.partition_search(net)

        data["cr"].append(result.stats.cr_core)
        data["time"].append(time.time() - start)
        data["error"].append(result.stats.re_f)
        print(
            "cr = {}, time = {}, error = {}".format(
                data["cr"][-1], data["time"][-1], data["error"][-1]
            )
        )

        # Get the best network
        net = result.best_state.network
        plt.clf()
        net.draw()
        plt.savefig(
            tn_diagrams_dir / (file_path.stem + f"_tnsseps{config.engine.eps}.png"),
            dpi=100,
            transparent=True,
        )
        plt.clf()

        # Save actions for best network
        with open(
            tn_actions_dir / (file_path.stem + f"_tnsseps{config.engine.eps}.pkl"), "wb"
        ) as f:
            pickle.dump(result.best_state.past_actions, f)

    return data


if __name__ == "__main__":
    # Truncation tolerances to consider
    epss = [float(f"1e-{i}") for i in range(8, 0, -1)]

    # TNSS config
    config = SearchConfig()
    config.engine.max_ops = 12
    config.engine.timeout = None
    config.engine.verbose = False
    config.synthesizer.action_type = "osplit"
    config.rank_search.error_split_stepsize = 1
    config.rank_search.k = 3

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

            # Make directory for TN diagrams
            os.makedirs(dir / subdir.split("/")[0] / "tn_diagrams", exist_ok=True)
            os.makedirs(dir / subdir.split("/")[0] / "tn_actions", exist_ok=True)

            # Get discretization data
            num_ordinates, factors = (
                (
                    [int(16 * 4**i) for i in range(8)],
                    [10 if dir != "fixed_source/quarter_circle" else 8],
                )
                if subdir == "direction/meshes"
                else ([256], np.geomspace(5, 100, 12).astype(int).tolist())
            )

            # Make list of files
            last_case = None
            for N, factor, degree, device in product(
                num_ordinates, factors, [2, 3, 4, 6], ["cpu", "gpu"]
            ):
                file_path = (
                    dir
                    / subdir
                    / "N{}_G1_A{}_B{}_p{}_q{}_eps1e-08{}.pkl".format(
                        N, factor + degree, factor + degree, degree, degree, device
                    )
                )
                if file_path.exists():
                    case = {
                        "num_ordinates": N,
                        "factor": factor,
                        "degree": degree,
                        "eps": 1e-8,
                        "solve_method": "CSR",
                    }
                    if case == last_case:
                        continue

                    # Set discretization
                    discretization = {
                        "$i_{q}$": 4,
                        r"$i_{\mu}$": int(np.sqrt(N / 4)),
                        r"$i_{\gamma}$": int(np.sqrt(N / 4)),
                        "$i_{E}$": 1,
                        "$i_{e}$": 1 if dir != dirs[-1] else 4,
                        r"$i_{\hat{x}}$": factor + degree,
                        r"$i_{\hat{y}}$": factor + degree,
                    }

                    # Run TN-SS
                    data = tnss(
                        config,
                        epss,
                        dir / subdir.split("/")[0] / "tn_diagrams",
                        dir / subdir.split("/")[0] / "tn_actions",
                        file_path,
                        discretization,
                    )
                    if data is None:
                        continue

                    with open(
                        dir / subdir.split("/")[0] / "tnss_data.jsonl",
                        "a",
                        encoding="utf-8",
                    ) as f:
                        # Serialize dict to string and dump data
                        f.write(
                            json.dumps(
                                {
                                    **case,
                                    "device": device,
                                    **data,
                                },
                                ensure_ascii=False,
                            )
                            + "\n"
                        )

                    # Update last case
                    last_case = case

    # =======================================================
    # Angular and mesh resolution studies
    # =======================================================
    files = [
        Path(file)
        for file in [
            "fixed_source/cruciform/data.pkl",
            "eigenvalue/circle/solutions.pkl",
            "eigenvalue/quarter_circle/data.pkl",
            "eigenvalue/pincell/data.pkl",
            "eigenvalue/lightbridge_ba/data.pkl",
            "eigenvalue/lightbridge_gas/solutrions.pkl",
        ]
    ]

    discretizations = [
        {
            "num_ordinates": num_ordinates,
            "num_patches": num_patches,
            "num_groups": num_groups,
            "factor": factor,
            "degree": degree,
            "eps": eps,
        }
        for num_ordinates, num_patches, num_groups, factor, degree, eps in zip(
            [4096, 4096, 4096, 1024, 1024, 1024],
            [12, 1, 1, 4, 12, 12],
            [1, 1, 1, 7, 7, 7],
            [13, 10, [10, 16], 10, 10, 10],
            [3, 4, 4, 2, 2, 2],
            6 * [1e-5],
        )
    ]

    for file, discretization in zip(files, discretizations):
        print("=" * 80 + "\nDirectory: {}".format(file.parent))

        # Make folder for TN diagrams
        os.makedirs(file.parent / "tn_diagrams", exist_ok=True)
        os.makedirs(file.parent / "tn_actions", exist_ok=True)

        # Run TN-SS
        data = tnss(
            config,
            epss,
            file.parent / "tn_diagrams",
            file.parent / "tn_actions",
            file,
            {
                "$i_{q}$": 4,
                r"$i_{\mu}$": int(np.sqrt(discretization["num_ordinates"] / 4)),
                r"$i_{\gamma}$": int(np.sqrt(discretization["num_ordinates"] / 4)),
                "$i_{E}$": discretization["num_groups"],
                "$i_{e}$": discretization["num_patches"],
                r"$i_{\hat{x}}$": (
                    discretization["factor"][0]
                    if isinstance(discretization["factor"], list)
                    else discretization["factor"]
                )
                + discretization["degree"],
                r"$i_{\hat{y}}$": (
                    discretization["factor"][1]
                    if isinstance(discretization["factor"], list)
                    else discretization["factor"]
                )
                + discretization["degree"],
            },
        )

        with open(file.parent / "tnss_stats.json", "w") as f:
            json.dump(
                {**discretization, "solve_method": "CSR", "device": "gpu", **data},
                f,
                indent=4,
            )
