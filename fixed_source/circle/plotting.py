import itertools
import json
import multiprocessing
import os
import pickle
import sys
from pathlib import Path
from typing import Tuple, Union

if __name__ == "__main__":
    multiprocessing.set_start_method("forkserver")
    sys.path.append("../..")

import matplotlib.pyplot as plt
import numpy as np
from extract import get_jsonl_data, get_pickle_data


def eps2str(eps):
    for i in range(10):
        if eps == float(f"1e-{i}"):
            return f"10^{i}"

    raise RuntimeError(f"Failed to find string for eps={eps}")


def prettyOp(op, format):
    if op == "B_in":
        return "B_{in}^{" + format + "}"
    if op == "B_out":
        return "B_{out}^{" + format + "}"

    return f"{op}^" + "{" + format + "}"


if __name__ == "__main__":
    # Path to this directory
    dir = Path(os.path.dirname(os.path.abspath(__file__)))

    # Make figure directory
    (dir / "direction/figs").mkdir(parents=True, exist_ok=True)

    # Solutions from OpenMC
    leakage_frac_openmc = [0.43995423399999983, 2.2245143201699137e-05]

    num_ordinates = [16, 64, 256, 1024, 4096, 16384, 65536, 262144]
    degrees = [2, 3, 4, 6]
    eps = [1e-8, 1e-5, 1e-3]

    # ========================================================================
    # Leakage Fraction Plots
    # ========================================================================

    # Extract leakage fraction data
    data = get_jsonl_data(
        dir / "direction/processed_direction.jsonl",
        lambda line_data: (
            (True, line_data["leakage_fraction"])
            if line_data["device"] == "cpu"
            else (False, None)
        ),
    )

    linestyles = ["-", "--", ":", "-."]
    markers = ["o", "s", "^", "D"]

    # Plot leakage fraction
    plt.hlines(
        [leakage_frac_openmc[0]],
        num_ordinates[0],
        num_ordinates[-1],
        label="OpenMC$\\pm 2\\sigma$",
        color="black",
    )
    plt.fill_between(
        num_ordinates,
        leakage_frac_openmc[0] - 2 * leakage_frac_openmc[-1],
        leakage_frac_openmc[0] + 2 * leakage_frac_openmc[-1],
        color="black",
        alpha=0.2,
    )
    for i, degree in enumerate(degrees):
        plt.plot(
            [
                d["num_ordinates"]
                for d in data
                if d["eps"] == eps[0] and d["degree"] == degree
            ],
            [
                d["value"][0]
                for d in data
                if d["eps"] == eps[0] and d["degree"] == degree
            ],
            "-o",
            label=f"CSR: $p = {degree}$",
        )
    plt.ylabel("Leakage Fraction")
    plt.xlabel("Number of Ordinates")
    plt.xscale("log")
    # plt.yscale("log")
    plt.legend()
    plt.tight_layout()
    plt.savefig("./direction/figs/openmc_vs_csr_lf.png", dpi=300)

    # Plot CSR leakage fraction Z-score to OpenMC
    plt.clf()
    plt.hlines(
        [2],
        num_ordinates[0],
        num_ordinates[-1],
        label="$2\\sigma$ to OpenMC",
        color="black",
    )
    plt.hlines(
        [1],
        num_ordinates[0],
        num_ordinates[-1],
        linestyles="--",
        label="$\\sigma$ to OpenMC",
        color="black",
    )
    plt.fill_between(
        num_ordinates,
        0,
        [2],
        color="black",
        alpha=0.2,
    )
    for i, degree in enumerate(degrees):
        plt.plot(
            [
                d["num_ordinates"]
                for d in data
                if d["eps"] == eps[0] and d["degree"] == degree
            ],
            [
                abs(d["zscore"][0])
                for d in data
                if d["eps"] == eps[0] and d["degree"] == degree
            ],
            "-o",
            label=f"CSR: $p = {degree}$",
        )
    plt.ylabel("$\\#$ of $\\sigma$ from OpenMC")
    plt.xlabel("Number of Ordinates")
    plt.xscale("log")
    plt.yscale("log")
    plt.legend(ncol=2)
    plt.tight_layout()
    plt.savefig("./direction/figs/openmc_vs_csr_lfz.png", dpi=300)

    # Plot CSR leakage fraction error to OpenMC
    plt.clf()
    plt.hlines(
        [(2 * leakage_frac_openmc[1]) / leakage_frac_openmc[0]],
        num_ordinates[0],
        num_ordinates[-1],
        label="$2\\sigma$ to OpenMC",
        color="black",
    )
    plt.hlines(
        [(leakage_frac_openmc[1]) / leakage_frac_openmc[0]],
        num_ordinates[0],
        num_ordinates[-1],
        linestyles="--",
        label="$\\sigma$ to OpenMC",
        color="black",
    )
    plt.fill_between(
        num_ordinates,
        0,
        [(2 * leakage_frac_openmc[1]) / leakage_frac_openmc[0]],
        color="black",
        alpha=0.2,
    )
    for i, degree in enumerate(degrees):
        plt.plot(
            [
                d["num_ordinates"]
                for d in data
                if d["eps"] == eps[0] and d["degree"] == degree
            ],
            np.array(
                [
                    abs(d["error"][0])
                    for d in data
                    if d["eps"] == eps[0] and d["degree"] == degree
                ]
            )
            / leakage_frac_openmc[0],
            "-o",
            label=f"CSR: $p = {degree}$",
        )
    plt.ylabel("Leakage Fraction Relative Error to OpenMC")
    plt.xlabel("Number of Ordinates")
    plt.xscale("log")
    plt.yscale("log")
    plt.legend(ncol=2)
    plt.tight_layout()
    plt.savefig("./direction/figs/openmc_vs_csr_lfe.png", dpi=300)

    # Look at errors relative to CSR
    plt.clf()
    for degree in degrees:
        # Get CSR solution
        csr = [
            np.array(
                [
                    d["num_ordinates"]
                    for d in data
                    if d["eps"] == eps[0] and d["degree"] == degree
                ]
            ),
            np.array(
                [
                    abs(d["value"][0])
                    for d in data
                    if d["eps"] == eps[0] and d["degree"] == degree
                ]
            ),
        ]

        # Iterate through eps
        plt.clf()
        for i in range(len(eps)):
            ordinates = [
                d["num_ordinates"]
                for d in data
                if len(d["value"]) > i and d["eps"] == eps[i] and d["degree"] == degree
            ]
            plt.plot(
                ordinates,
                np.abs(
                    np.array(
                        [
                            abs(d["value"][-1])
                            for d in data
                            if len(d["value"]) > i
                            and d["eps"] == eps[i]
                            and d["degree"] == degree
                        ]
                    )
                    - csr[1][: len(ordinates)]
                )
                / csr[1][: len(ordinates)],
                "-o",
                label=f"TT (rounded): $p={degree},\\epsilon={eps2str(eps[i])}$",
            )

        plt.xlabel("Number of Ordinates")
        plt.ylabel("Leakage Fraction Relative Error to CSR")
        plt.xscale("log")
        plt.yscale("log")
        plt.legend()
        plt.tight_layout()
        plt.savefig(f"./direction/figs/tt_vs_csr_p{degree}_lfe.png", dpi=300)

    # ========================================================================
    # Ranks and Compression
    # ========================================================================

    # Get ranks for all operators
    data = get_jsonl_data(
        dir / "direction/processed_direction.jsonl",
        lambda line_data: (
            (True, line_data["ranks"])
            if line_data["device"] == "cpu" and "ranks" in line_data
            else (False, None)
        ),
    )

    plt.clf()
    for op in ["H", "S", "B_out", "T"]:
        for degree in degrees:
            plt.clf()
            for i in range(len(eps)):
                plt.plot(
                    [
                        d["num_ordinates"]
                        for d in data
                        if op in d and d["eps"] == eps[i] and d["degree"] == degree
                    ],
                    [
                        np.array(d[op]).max()
                        for d in data
                        if op in d and d["eps"] == eps[i] and d["degree"] == degree
                    ],
                    "-o",
                    label=f"$p={degree},\\epsilon={eps2str(eps[i])}$",
                )
            plt.xlabel("Number of Ordinates")
            plt.ylabel(f"Max Rank of ${prettyOp(op, 'TT')}$")
            plt.xscale("log")
            plt.legend()
            plt.tight_layout()
            plt.savefig(f"./direction/figs/{op}_max_rank_p{degree}.png", dpi=300)

    data = get_jsonl_data(
        dir / "direction/processed_direction.jsonl",
        lambda line_data: (
            (True, line_data["compression"]["tts"])
            if line_data["device"] == "cpu" and "tts" in line_data["compression"]
            else (False, None)
        ),
    )

    data2 = get_jsonl_data(
        dir / "direction/processed_direction.jsonl",
        lambda line_data: (
            (True, line_data["compression"]["matrix"])
            if line_data["device"] == "cpu" and "matrix" in line_data["compression"]
            else (False, None)
        ),
    )

    plt.clf()
    for op in ["H", "S", "B_out"]:
        for degree in degrees:
            plt.clf()
            plt.plot(
                [
                    d["num_ordinates"]
                    for d in data2
                    if d["eps"] == eps[0] and d["degree"] == degree
                ],
                [
                    np.array(d[op]).max()
                    for d in data2
                    if d["eps"] == eps[0] and d["degree"] == degree
                ],
                "-o",
                label=f"CSR: $p={degree}$",
            )
            for i in range(len(eps)):
                plt.plot(
                    [
                        d["num_ordinates"]
                        for d in data
                        if d["eps"] == eps[i] and d["degree"] == degree
                    ],
                    [
                        np.array(d[op]).max()
                        for d in data
                        if d["eps"] == eps[i] and d["degree"] == degree
                    ],
                    "-o",
                    label=f"TT: $p={degree},\\epsilon={eps2str(eps[i])}$",
                )
            plt.xlabel("Number of Ordinates")
            plt.ylabel(f"Compression of ${prettyOp(op, 'TT')}$")
            plt.xscale("log")
            plt.yscale("log")
            plt.legend()
            plt.tight_layout()
            plt.savefig(f"./direction/figs/{op}_compression_p{degree}.png", dpi=300)

    # data = get_jsonl_data(
    #     dir / "direction/processed_direction.jsonl",
    #     lambda line_data: (
    #         (True, {"total", line_data["compression"]["total"]})
    #         if line_data["device"] == "cpu"
    #         else (False, None)
    #     ),
    # )

    # TODO: Rerun to make sure we actually get this for all operators
    # # SVD truncation tolerance doesn't actually change the number of ranks
    # clf.eps()
    # for degree in degrees:
    #     for i, name in enumerate(["CSR", "TT", "Mixed", "TT (rounded)"]):
    #         plt.plot(
    #             [
    #                 d["num_ordinates"]
    #                 for d in data
    #                 if d["eps"] == eps[0] and d["degree"] == degree
    #             ],
    #             [
    #                 np.array(d[total]).max()
    #                 for d in data
    #                 if d["eps"] == eps[i] and d["degree"] == degree
    #             ],
    #         )

    # ========================================================================
    # Matvec scaling
    # ========================================================================

    # Get Matvec information, In this case we don't care about eps as it had no effect
    data = get_jsonl_data(
        dir / "direction/processed_direction.jsonl",
        lambda line_data: (
            (True, {"solve_method": line_data["solve_method"], **line_data["matvec"]})
            if line_data["eps"] == eps[0]
            else (False, None)
        ),
    )

    for degree in degrees:
        plt.clf()
        for device in ["cpu", "gpu"]:
            for i, case in enumerate(["CSR", "TT", "Mixed", "TT (rounded)"]):
                plt.plot(
                    [
                        d["num_ordinates"]
                        for d in data
                        if case in d["solve_method"]
                        and d["eps"] == eps[0]
                        and d["degree"] == degree
                        and d["device"] == device
                    ],
                    np.array(
                        [
                            d["time"][i]
                            for d in data
                            if case in d["solve_method"]
                            and d["eps"] == eps[0]
                            and d["degree"] == degree
                            and d["device"] == device
                        ]
                    )
                    * 1000,
                    "-o",
                    label=f"{case}: $p={degree}$,{device.upper()}",
                )

        plt.xlabel("Number of Ordinates")
        plt.ylabel(f"Average SpMV Time (ms)")
        plt.xscale("log")
        plt.yscale("log")
        plt.legend()
        plt.tight_layout()
        plt.savefig(f"./direction/figs/matvec_p{degree}.png", dpi=300)

    # ========================================================================
    # GMRES scaling
    # ========================================================================

    # Get GMRES information, In this case we don't care about eps as it had no effect
    data = get_jsonl_data(
        dir / "direction/processed_direction.jsonl",
        lambda line_data: (
            (True, {"solve_method": line_data["solve_method"], **line_data["gmres"]})
            if line_data["eps"] == eps[0]
            else (False, None)
        ),
    )

    for degree in degrees:
        plt.clf()
        for device in ["cpu", "gpu"]:
            for i, case in enumerate(["CSR", "TT", "Mixed", "TT (rounded)"]):
                plt.plot(
                    [
                        d["num_ordinates"]
                        for d in data
                        if case in d["solve_method"]
                        and d["eps"] == eps[0]
                        and d["degree"] == degree
                        and d["device"] == device
                    ],
                    np.array(
                        [
                            d["time"][i]
                            for d in data
                            if case in d["solve_method"]
                            and d["eps"] == eps[0]
                            and d["degree"] == degree
                            and d["device"] == device
                        ]
                    ),
                    "-o",
                    label=f"{case}: $p={degree}$, {device.upper()}",
                )

        plt.xlabel("Number of Ordinates")
        plt.ylabel(f"GMRES Run Time (s)")
        plt.xscale("log")
        plt.yscale("log")
        plt.legend()
        plt.tight_layout()
        plt.savefig(f"./direction/figs/gmres_time_p{degree}.png", dpi=300)
