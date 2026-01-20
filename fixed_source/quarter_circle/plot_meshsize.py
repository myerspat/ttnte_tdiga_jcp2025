import os
import sys
from pathlib import Path

sys.path.append("../..")

import matplotlib.pyplot as plt
import numpy as np
from scipy import stats

from extract import get_jsonl_data

# Change plotting label sizes
plt.rcParams["font.size"] = 14
plt.rcParams["axes.labelsize"] = 16
plt.rcParams["xtick.labelsize"] = 14
plt.rcParams["ytick.labelsize"] = 14
plt.rcParams["legend.fontsize"] = 14
plt.rcParams["axes.grid"] = True


def eps2str(eps):
    for i in range(10):
        if eps == float(f"1e-{i}"):
            return "10^{" + str(-i) + "}"

    raise RuntimeError(f"Failed to find string for eps={eps}")


def prettyOp(op, format=None):
    name = r"\mathcal{" + op + "}"
    if op == "B_in":
        name = r"\mathcal{B}_{\text{in}}"
    if op == "B_out":
        name = r"\mathcal{B}_{\text{out}}"

    if format is not None:
        name += "^{" + format + "}"

    return name


if __name__ == "__main__":
    # Path to this directory
    dir = Path(os.path.dirname(os.path.abspath(__file__)))

    # Make figure directory
    (dir / "meshsize/figs").mkdir(parents=True, exist_ok=True)

    # Solutions from OpenMC
    leakage_frac_openmc = [0.43995486599999994, 2.2442114486922458e-05]
    phi_mc = np.load(
        "../../../ttnte/notebooks/fixed_source/quarter_circle/openmc/data/mesh_flux.npy"
    )
    phi_mc_stdev = np.load(
        "../../../ttnte/notebooks/fixed_source/quarter_circle/openmc/data/mesh_stdev.npy"
    )

    # Discretization
    num_ordinates = [256]
    factor = np.geomspace(5, 100, 12).astype(int).tolist()
    degrees = [2, 3, 4, 6]
    eps = [1e-8, 1e-5, 1e-3]

    linestyles = ["-", "--", ":", "-."]
    markers = ["o", "s", "^", "D"]
    colors = [
        "#0072B2",
        "#E69F00",
        "#56B4E9",
        "#D55E00",
        "#009E73",
        "#F0E442",
        "#000000",
    ]

    # ========================================================================
    # Leakage Fraction Plots
    # ========================================================================

    # Extract leakage fraction data
    data = get_jsonl_data(
        dir / "meshsize/processed_meshsize.jsonl",
        lambda line_data: (
            (True, line_data["leakage_fraction"])
            if line_data["device"] == "gpu"
            else (False, None)
        ),
    )

    # Plot leakage fraction
    plt.clf()
    plt.hlines(
        [leakage_frac_openmc[0]],
        4 * (factor[0] + degrees[0]) ** 2,
        4 * (factor[-1] + degrees[-1]) ** 2,
        label=r"$f_{\text{Leak}}^{\text{ref}}\pm 2\sigma$",
        color="black",
    )
    plt.fill_between(
        [4 * (factor[0] + degrees[0]) ** 2, 4 * (factor[-1] + degrees[-1]) ** 2],
        leakage_frac_openmc[0] - 2 * leakage_frac_openmc[-1],
        leakage_frac_openmc[0] + 2 * leakage_frac_openmc[-1],
        color="black",
        alpha=0.2,
    )
    for i, degree in enumerate(degrees):
        plt.plot(
            [
                4 * (d["factor"] + d["degree"]) ** 2
                for d in data
                if d["eps"] == eps[0]
                and d["degree"] == degree
                and "CSR" in d["solve_method"]
            ],
            [
                d["value"][
                    np.argwhere(np.array(d["solve_method"]) == "CSR").flatten()[0]
                ]
                for d in data
                if d["eps"] == eps[0]
                and d["degree"] == degree
                and "CSR" in d["solve_method"]
            ],
            "-o",
            label=r"$p_{\hat{x}} = p_{\hat{y}} = " + f"{degree}$",
        )
    plt.ylabel(r"$f_{\text{leak}}^{\text{CSR}}$")
    plt.xlabel(r"$N_e \times N_{\hat{x}} \times N_{\hat{y}}$")
    plt.xscale("log")
    plt.legend()
    plt.tight_layout()
    plt.savefig("./meshsize/figs/leakage.jpeg", dpi=700, transparent=False)

    # Plot CSR leakage fraction error to OpenMC
    plt.clf()
    plt.hlines(
        [(2 * leakage_frac_openmc[1]) / leakage_frac_openmc[0]],
        4 * (factor[0] + degrees[0]) ** 2,
        4 * (factor[-1] + degrees[-1]) ** 2,
        label=r"$f_{\text{leak}}^{\text{X}} = f_{\text{leak}}^{\text{MC}} + 2\sigma$",
        color="black",
    )
    plt.hlines(
        [(leakage_frac_openmc[1]) / leakage_frac_openmc[0]],
        4 * (factor[0] + degrees[0]) ** 2,
        4 * (factor[-1] + degrees[-1]) ** 2,
        linestyles="--",
        label=r"$f_{\text{leak}}^{\text{X}} = f_{\text{leak}}^{\text{MC}} + \sigma$",
        color="black",
    )
    plt.fill_between(
        [4 * (factor[0] + degrees[0]) ** 2, 4 * (factor[-1] + degrees[-1]) ** 2],
        0,
        [(2 * leakage_frac_openmc[1]) / leakage_frac_openmc[0]],
        color="black",
        alpha=0.2,
    )
    for i, degree in enumerate(degrees):
        plt.plot(
            [
                4 * (d["factor"] + d["degree"]) ** 2
                for d in data
                if d["eps"] == eps[0]
                and d["degree"] == degree
                and "CSR" in d["solve_method"]
            ],
            np.array(
                [
                    abs(
                        d["error"][
                            np.argwhere(np.array(d["solve_method"]) == "CSR").flatten()[
                                0
                            ]
                        ]
                    )
                    for d in data
                    if d["eps"] == eps[0]
                    and d["degree"] == degree
                    and "CSR" in d["solve_method"]
                ]
            )
            / leakage_frac_openmc[0],
            "-o",
            label=r"$p_{\hat{x}} = p_{\hat{y}} = " + f"{degree}$",
        )
    plt.ylabel(
        r"$\delta f\left(f_{\text{leak}}^{\text{CSR}}, f_{\text{leak}}^{\text{MC}}\right)$"
    )
    plt.xlabel(r"$N_e \times N_{\hat{x}} \times N_{\hat{y}}$")
    plt.xscale("log")
    plt.yscale("log")
    plt.legend()
    plt.tight_layout()
    plt.savefig("./meshsize/figs/leakage_relerror.jpeg", dpi=700, transparent=False)

    # Look at errors relative to CSR
    plt.clf()
    for degree in degrees:
        # Get CSR solution
        csr = [
            np.array(
                [
                    4 * (d["factor"] + d["degree"]) ** 2
                    for d in data
                    if d["eps"] == eps[0]
                    and d["degree"] == degree
                    and "CSR" in d["solve_method"]
                ],
            ),
            np.array(
                [
                    abs(
                        d["value"][
                            np.argwhere(np.array(d["solve_method"]) == "CSR").flatten()[
                                0
                            ]
                        ]
                    )
                    for d in data
                    if d["eps"] == eps[0]
                    and d["degree"] == degree
                    and "CSR" in d["solve_method"]
                ]
            ),
        ]
        # Iterate through eps
        for solve_method in ["Mixed", "Mixed (rounded)"]:
            plt.clf()
            for i in range(len(eps)):
                method = np.array(
                    [
                        np.array(
                            [
                                4 * (d["factor"] + d["degree"]) ** 2
                                for d in data
                                if d["eps"] == eps[i]
                                and d["degree"] == degree
                                and solve_method in d["solve_method"]
                            ],
                        ),
                        np.array(
                            [
                                abs(
                                    d["value"][
                                        np.argwhere(
                                            np.array(d["solve_method"]) == solve_method
                                        ).flatten()[0]
                                    ]
                                )
                                for d in data
                                if d["eps"] == eps[i]
                                and d["degree"] == degree
                                and solve_method in d["solve_method"]
                            ]
                        ),
                    ]
                )
                plt.plot(
                    method[0, np.isin(method[0], csr[0])],
                    np.abs(
                        method[1, np.isin(method[0], csr[0])]
                        - csr[1][np.isin(csr[0], method[0])]
                    )
                    / csr[1][np.isin(csr[0], method[0])],
                    "-o",
                    label=rf"$\epsilon={eps2str(eps[i])}$",
                )

            plt.xlabel(r"$N_e \times N_{\hat{x}} \times N_{\hat{y}}$")
            plt.ylabel(
                r"$\delta f\left(f_{\text{leak}}^{\text{"
                + solve_method
                + r"}}, f_{\text{leak}}^{\text{CSR}}\right)$"
            )
            plt.xscale("log")
            plt.yscale("log")
            plt.legend(fontsize=14)
            plt.tight_layout()
            plt.savefig(
                f"./meshsize/figs/leakage_relerror_p{degree}_{solve_method}.eps",
                dpi=700,
                transparent=False,
            )

    # ========================================================================
    # Ranks and Compression
    # ========================================================================

    # Get ranks for all operators
    data = get_jsonl_data(
        dir / "meshsize/processed_meshsize.jsonl",
        lambda line_data: (
            (True, line_data["ranks"])
            if line_data["device"] == "gpu"
            else (False, None)
        ),
    )

    plt.clf()
    for op in ["H", "S", "B_out", "B_in"]:
        for degree in degrees:
            plt.clf()
            for i in range(len(eps)):
                plt.plot(
                    [
                        4 * (d["factor"] + d["degree"]) ** 2
                        for d in data
                        if d["eps"] == eps[i] and d["degree"] == degree
                    ],
                    [
                        np.array(d[op]).max()
                        for d in data
                        if op in d and d["eps"] == eps[i] and d["degree"] == degree
                    ],
                    "-o",
                    label=rf"$\epsilon={eps2str(eps[i])}$",
                )
            plt.xlabel(r"$N_e \times N_{\hat{x}} \times N_{\hat{y}}$")
            plt.ylabel(r"$r_{\text{max}}\left(" + prettyOp(op, "TT") + r"\right)$")
            plt.xscale("log")
            plt.legend()
            plt.tight_layout()
            plt.savefig(
                f"./meshsize/figs/ranks_p{degree}_{op}.eps", dpi=700, transparent=False
            )

    data = get_jsonl_data(
        dir / "meshsize/processed_meshsize.jsonl",
        lambda line_data: (
            (True, line_data["compression"]["tts"])
            if line_data["device"] == "gpu"
            else (False, None)
        ),
    )
    data2 = get_jsonl_data(
        dir / "meshsize/processed_meshsize.jsonl",
        lambda line_data: (
            (True, line_data["compression"]["matrix"])
            if line_data["device"] == "gpu" and line_data["eps"] == eps[0]
            else (False, None)
        ),
    )

    plt.clf()
    for op in ["H", "S", "B_out", "B_in"]:
        print(op)
        for degree in degrees:
            print(f"degree = {degree}")
            plt.clf()
            plt.plot(
                [
                    4 * (d["factor"] + d["degree"]) ** 2
                    for d in data2
                    if d["degree"] == degree
                ],
                [np.array(d[op]).max() for d in data2 if d["degree"] == degree],
                "--o",
                label=r"CSR",
            )
            csr_data = [
                np.array(
                    [
                        4 * (d["factor"] + d["degree"]) ** 2
                        for d in data2
                        if d["degree"] == degree
                    ]
                )[:3],
                np.array(
                    [np.array(d[op]).max() for d in data2 if d["degree"] == degree]
                )[:3],
            ]
            print(
                "Slope = {}, intercept = {}, r_value = {}, p_value = {}, std_err = {}".format(
                    *stats.linregress(np.log10(csr_data[0]), np.log10(csr_data[1]))
                )
            )
            csr_data = [
                np.array(
                    [
                        4 * (d["factor"] + d["degree"]) ** 2
                        for d in data2
                        if d["degree"] == degree
                    ]
                )[-3:],
                np.array(
                    [np.array(d[op]).max() for d in data2 if d["degree"] == degree]
                )[-3:],
            ]
            print(
                "Slope = {}, intercept = {}, r_value = {}, p_value = {}, std_err = {}".format(
                    *stats.linregress(np.log10(csr_data[0]), np.log10(csr_data[1]))
                ),
            )
            for i in range(len(eps)):
                print(f"eps = {eps[i]}")
                tt_data = [
                    np.array(
                        [
                            4 * (d["factor"] + d["degree"]) ** 2
                            for d in data
                            if d["eps"] == eps[i] and d["degree"] == degree
                        ]
                    )[:3],
                    np.array(
                        [
                            np.array(d[op]).max()
                            for d in data
                            if d["eps"] == eps[i] and d["degree"] == degree
                        ]
                    )[:3],
                ]
                print(
                    "Slope = {}, intercept = {}, r_value = {}, p_value = {}, std_err = {}".format(
                        *stats.linregress(np.log10(tt_data[0]), np.log10(tt_data[1]))
                    )
                )
                tt_data = [
                    np.array(
                        [
                            4 * (d["factor"] + d["degree"]) ** 2
                            for d in data
                            if d["eps"] == eps[i] and d["degree"] == degree
                        ]
                    )[-3:],
                    np.array(
                        [
                            np.array(d[op]).max()
                            for d in data
                            if d["eps"] == eps[i] and d["degree"] == degree
                        ]
                    )[-3:],
                ]
                print(
                    "Slope = {}, intercept = {}, r_value = {}, p_value = {}, std_err = {}".format(
                        *stats.linregress(np.log10(tt_data[0]), np.log10(tt_data[1]))
                    ),
                )

                plt.plot(
                    [
                        4 * (d["factor"] + d["degree"]) ** 2
                        for d in data
                        if d["eps"] == eps[i] and d["degree"] == degree
                    ],
                    [
                        np.array(d[op]).max()
                        for d in data
                        if d["eps"] == eps[i] and d["degree"] == degree
                    ],
                    "-o",
                    label=rf"TT, $\epsilon={eps2str(eps[i])}$",
                )
            print()
            plt.xlabel(r"$N_e \times N_{\hat{x}} \times N_{\hat{y}}$")
            plt.ylabel(r"$\text{CR}\left(" + prettyOp(op) + r"\right)$")
            plt.xscale("log")
            plt.yscale("log")
            plt.legend()
            plt.tight_layout()
            plt.savefig(
                f"./meshsize/figs/compression_p{degree}_{op}.eps",
                dpi=700,
                transparent=False,
            )

            plt.clf()
            csr = [
                np.array(
                    [
                        4 * (d["factor"] + d["degree"]) ** 2
                        for d in data2
                        if d["degree"] == degree
                    ]
                ),
                np.array(
                    [np.array(d[op]).max() for d in data2 if d["degree"] == degree]
                ),
            ]
            for i in range(len(eps)):
                tt = [
                    np.array(
                        [
                            4 * (d["factor"] + d["degree"]) ** 2
                            for d in data
                            if d["eps"] == eps[i] and d["degree"] == degree
                        ]
                    ),
                    np.array(
                        [
                            np.array(d[op]).max()
                            for d in data
                            if d["eps"] == eps[i] and d["degree"] == degree
                        ]
                    ),
                ]
                plt.plot(
                    tt[0][np.isin(tt[0], csr[0])],
                    tt[1][np.isin(tt[0], csr[0])] / csr[1][np.isin(csr[0], tt[0])],
                    "-o",
                    label=rf"$\epsilon={eps2str(eps[i])}$",
                )
            plt.xlabel(r"$N_e \times N_{\hat{x}} \times N_{\hat{y}}$")
            plt.ylabel(
                r"$\text{CR}\left("
                + prettyOp(op, "TT")
                + r"\right)/\text{CR}\left("
                + prettyOp(op, "CSR")
                + r"\right)$"
            )
            plt.xscale("log")
            plt.yscale("log")
            plt.legend()
            plt.tight_layout()
            plt.savefig(
                f"./meshsize/figs/compression_ratio_p{degree}_{op}.eps",
                dpi=700,
                transparent=False,
            )

    data = get_jsonl_data(
        dir / "meshsize/processed_meshsize.jsonl",
        lambda line_data: (
            (True, {"total": line_data["compression"]["total"]})
            if line_data["device"] == "gpu"
            else (False, None)
        ),
    )

    for degree in degrees:
        for i in range(len(eps)):
            plt.clf()
            plt.plot(
                [
                    4 * (d["factor"] + d["degree"]) ** 2
                    for d in data
                    if d["eps"] == eps[0]
                    and d["degree"] == degree
                    and "CSR" in d["solve_method"]
                ],
                [
                    d["total"][
                        np.argwhere(np.array(d["solve_method"]) == "CSR").flatten()[0]
                    ]
                    for d in data
                    if d["eps"] == eps[0]
                    and d["degree"] == degree
                    and "CSR" in d["solve_method"]
                ],
                "-o",
                color=colors[0],
                label="CSR",
            )

            for j, name in enumerate(["Mixed", "Mixed (rounded)"]):
                plt.plot(
                    [
                        4 * (d["factor"] + d["degree"]) ** 2
                        for d in data
                        if d["eps"] == eps[i]
                        and d["degree"] == degree
                        and name in d["solve_method"]
                    ],
                    [
                        d["total"][
                            np.argwhere(np.array(d["solve_method"]) == name).flatten()[
                                0
                            ]
                        ]
                        for d in data
                        if d["eps"] == eps[i]
                        and d["degree"] == degree
                        and name in d["solve_method"]
                    ],
                    "-o",
                    color=colors[j + 1],
                    label=name,
                )

            plt.xlabel(r"$N_e \times N_{\hat{x}} \times N_{\hat{y}}$")
            plt.ylabel(r"$\text{CR}\left(" + prettyOp("T") + r"\right)$")
            plt.xscale("log")
            plt.yscale("log")
            plt.legend()
            plt.tight_layout()
            plt.savefig(
                f"./meshsize/figs/compression_p{degree}_eps{eps[i]}_T.eps",
                dpi=700,
                transparent=False,
            )

    # Get angular flux compression ranks and compression
    data = get_jsonl_data(
        dir / "meshsize/processed_meshsize.jsonl",
        lambda line_data: (
            (
                True,
                {
                    "ranks": line_data["flux_stats"]["ranks"],
                    "compression": line_data["flux_stats"]["compression"],
                },
            )
            if line_data["device"] == "gpu"
            else (False, None)
        ),
    )

    for degree in degrees:
        plt.clf()
        for i in range(len(eps)):
            if i == 0:
                num_ords = np.array(
                    [
                        4 * (d["factor"] + d["degree"]) ** 2
                        for d in data
                        if "CSR" in d["solve_method"]
                        and d["eps"] == eps[0]
                        and d["degree"] == degree
                    ]
                )
                ranks = np.array(
                    [
                        np.array(d["ranks"][0]).max()
                        for d in data
                        if "CSR" in d["solve_method"]
                        and d["eps"] == eps[0]
                        and d["degree"] == degree
                    ]
                )
                plt.plot(
                    num_ords,
                    ranks,
                    "-o",
                    color=colors[-1],
                    label=r"$\epsilon = 0$",
                )
            num_ords = np.array(
                [
                    4 * (d["factor"] + d["degree"]) ** 2
                    for d in data
                    if d["eps"] == eps[i] and d["degree"] == degree
                ]
            )
            ranks = np.array(
                [
                    np.array(
                        d["ranks"][(1 if "CSR" in d["solve_method"] else 0) :]
                    ).max()
                    for d in data
                    if d["eps"] == eps[i] and d["degree"] == degree
                ]
            )
            plt.plot(
                num_ords,
                ranks,
                "--o",
                color=colors[i],
                label=rf"$\epsilon = {eps2str(eps[i])}$",
            )

        plt.xlabel(r"$N_e \times N_{\hat{x}} \times N_{\hat{y}}$")
        plt.ylabel(r"$r_{\text{max}}\left(\mathbf{\Psi}\right)$")
        plt.xscale("log")
        plt.yscale("log")
        plt.legend(fontsize=14)
        plt.tight_layout()
        plt.savefig(
            f"./meshsize/figs/ranks_p{degree}_psi.eps", dpi=700, transparent=False
        )

    for degree in degrees:
        plt.clf()
        for i in range(len(eps)):
            if i == 0:
                num_ords = np.array(
                    [
                        4 * (d["factor"] + d["degree"]) ** 2
                        for d in data
                        if "CSR" in d["solve_method"]
                        and d["eps"] == eps[0]
                        and d["degree"] == degree
                    ]
                )
                compression = np.array(
                    [
                        d["compression"][0]
                        for d in data
                        if "CSR" in d["solve_method"]
                        and d["eps"] == eps[0]
                        and d["degree"] == degree
                    ]
                )
                plt.plot(
                    num_ords,
                    compression,
                    "-o",
                    color=colors[-1],
                    label=r"$\epsilon = 0$",
                )
            num_ords = np.array(
                [
                    4 * (d["factor"] + d["degree"]) ** 2
                    for d in data
                    if d["eps"] == eps[i] and d["degree"] == degree
                ]
            )
            compression = np.array(
                [
                    np.array(
                        d["compression"][(1 if "CSR" in d["solve_method"] else 0) :]
                    ).max()
                    for d in data
                    if d["eps"] == eps[i] and d["degree"] == degree
                ]
            )
            plt.plot(
                num_ords,
                compression,
                "--o",
                color=colors[i],
                label=rf"$\epsilon = {eps2str(eps[i])}$",
            )

        plt.xlabel(r"$N_e \times N_{\hat{x}} \times N_{\hat{y}}$")
        plt.ylabel(r"$\text{CR}\left(\mathbf{\Psi}^{\text{TT}}\right)$")
        plt.xscale("log")
        plt.yscale("log")
        plt.legend(fontsize=14)
        plt.tight_layout()
        plt.savefig(
            f"./meshsize/figs/compression_p{degree}_psi.eps", dpi=700, transparent=False
        )

    # ========================================================================
    # Matvec scaling
    # ========================================================================

    # Get Matvec information, In this case we don't care about eps as it had no effect
    data = get_jsonl_data(
        dir / "meshsize/processed_meshsize.jsonl",
        lambda line_data: (
            (True, line_data["matvec"]) if line_data["eps"] == eps[0] else (False, None)
        ),
    )

    for degree in degrees:
        plt.clf()
        for device in ["gpu"]:
            for i, case in enumerate(["CSR", "Mixed", "Mixed (rounded)"]):
                num_ords = np.array(
                    [
                        (d["factor"] + d["degree"]) ** 2
                        for d in data
                        if case in d["solve_method"]
                        and (d["eps"] == eps[0] if case == "CSR" else eps[0])
                        and d["degree"] == degree
                        and d["device"] == device
                    ]
                )
                time = (
                    np.array(
                        [
                            d["time"][
                                np.argwhere(
                                    np.array(d["solve_method"]) == case
                                ).flatten()[0]
                            ]
                            for d in data
                            if case in d["solve_method"]
                            and (d["eps"] == eps[0] if case == "CSR" else eps[0])
                            and d["degree"] == degree
                            and d["device"] == device
                        ]
                    )
                    * 1000
                )
                stdev = (
                    np.array(
                        [
                            d["stdev"][
                                np.argwhere(
                                    np.array(d["solve_method"]) == case
                                ).flatten()[0]
                            ]
                            for d in data
                            if case in d["solve_method"]
                            and (d["eps"] == eps[0] if case == "CSR" else eps[0])
                            and d["degree"] == degree
                            and d["device"] == device
                        ]
                    )
                    * 1000
                )
                meshsize = (np.array(factor) + degree) ** 2
                plt.plot(
                    4
                    * num_ords[
                        np.isin(
                            num_ords,
                            np.array(meshsize)[(2 if degree == 2 else 0) :],
                        )
                    ],
                    time[
                        np.isin(
                            num_ords,
                            np.array(meshsize)[(2 if degree == 2 else 0) :],
                        )
                    ],
                    ("-" if device == "cpu" else "--") + "o",
                    color=colors[i],
                    label=f"{case}, {device.upper()}",
                )
                plt.fill_between(
                    4
                    * num_ords[
                        np.isin(
                            num_ords,
                            np.array(meshsize)[(2 if degree == 2 else 0) :],
                        )
                    ],
                    (time - stdev)[
                        np.isin(
                            num_ords,
                            np.array(meshsize)[(2 if degree == 2 else 0) :],
                        )
                    ],
                    (time + stdev)[
                        np.isin(
                            num_ords,
                            np.array(meshsize)[(2 if degree == 2 else 0) :],
                        )
                    ],
                    color=colors[i],
                    alpha=0.5,
                )

        plt.xlabel(r"$N_e \times N_{\hat{x}} \times N_{\hat{y}}$")
        plt.ylabel(r"Average SpMV Time $(ms)$")
        plt.xscale("log")
        plt.yscale("log")
        plt.legend(ncol=2, fontsize=10)
        plt.tight_layout()
        plt.savefig(
            f"./meshsize/figs/matvec_time_p{degree}_eps{eps[0]}.jpeg",
            dpi=700,
            transparent=False,
        )

    # ========================================================================
    # GMRES scaling
    # ========================================================================

    # Get GMRES information, In this case we don't care about eps as it had no effect
    data = get_jsonl_data(
        dir / "meshsize/processed_meshsize.jsonl",
        lambda line_data: (
            (True, line_data["gmres"]) if line_data["eps"] == eps[0] else (False, None)
        ),
    )

    for degree in degrees:
        plt.clf()
        for device in ["gpu"]:
            for i, case in enumerate(["CSR", "Mixed", "Mixed (rounded)"]):
                num_ords = np.array(
                    [
                        4 * (d["factor"] + d["degree"]) ** 2
                        for d in data
                        if case in d["solve_method"]
                        and (d["eps"] == eps[0] if case == "CSR" else eps[0])
                        and d["degree"] == degree
                        and d["device"] == device
                    ]
                )
                time = (
                    np.array(
                        [
                            d["time"][
                                np.argwhere(
                                    np.array(d["solve_method"]) == case
                                ).flatten()[0]
                            ]
                            for d in data
                            if case in d["solve_method"]
                            and (d["eps"] == eps[0] if case == "CSR" else eps[0])
                            and d["degree"] == degree
                            and d["device"] == device
                        ]
                    )
                    * 1000
                )
                plt.plot(
                    num_ords,
                    time,
                    ("-" if device == "cpu" else "--") + "o",
                    color=colors[i],
                    label=f"{case}, {device.upper()}",
                )

        plt.xlabel(r"$N_e \times N_{\hat{x}} \times N_{\hat{y}}$")
        plt.ylabel(r"GMRES Run Time $(s)$")
        plt.xscale("log")
        plt.yscale("log")
        plt.legend(ncol=2, fontsize=10)
        plt.tight_layout()
        plt.savefig(
            f"./meshsize/figs/gmres_time_p{degree}_eps{eps[0]}.eps",
            dpi=700,
            transparent=False,
        )

    # ========================================================================
    # L2-Error, zscore, etc
    # ========================================================================

    # Get flux information, only need 1e-8 eps as all operators are the same
    data = get_jsonl_data(
        dir / "meshsize/processed_meshsize.jsonl",
        lambda line_data: (
            (
                True,
                {
                    **line_data["flux_stats"],
                    "converged": line_data["gmres"]["converged"],
                },
            )
            if line_data["eps"] == eps[0] and line_data["device"] == "gpu"
            else (False, None)
        ),
    )

    plt.clf()
    # plt.hlines(
    #     [
    #         np.linalg.norm(2 * phi_mc_stdev.flatten(), 2)
    #         / np.linalg.norm(phi_mc.flatten(), 2)
    #     ],
    #     (factor[0] + degrees[0]) ** 2,
    #     (factor[-1] + degrees[-1]) ** 2,
    #     label=r"$\mathbf{\Phi}^{\text{X}} = \mathbf{\Phi}^{\text{MC}} + 2\boldsymbol{\sigma}^{\text{MC}}$",
    #     color="black",
    # )
    # plt.hlines(
    #     [
    #         np.linalg.norm(phi_mc_stdev.flatten(), 2)
    #         / np.linalg.norm(phi_mc.flatten(), 2)
    #     ],
    #     num_ordinates[0],
    #     num_ordinates[-1],
    #     linestyles="--",
    #     label=r"$\boldsymbol{\phi}^{\text{ref}}\pm \sigma^{\text{ref}}$",
    #     color="black",
    # )
    # plt.fill_between(
    #     [(factor[0] + degrees[0]) ** 2, (factor[-1] + degrees[-1]) ** 2],
    #     0,
    #     [
    #         np.linalg.norm(2 * phi_mc_stdev.flatten(), 2)
    #         / np.linalg.norm(phi_mc.flatten(), 2)
    #     ],
    #     color="black",
    #     alpha=0.2,
    # )
    print(
        np.linalg.norm(2 * phi_mc_stdev.flatten(), 2)
        / np.linalg.norm(phi_mc.flatten(), 2)
    )
    for i, degree in enumerate(degrees):
        plt.plot(
            [
                4 * (d["factor"] + d["degree"]) ** 2
                for d in data
                if d["eps"] == eps[0] and d["degree"] == degree
            ],
            np.array(
                [
                    d["l2 error"][0][0]
                    for d in data
                    if d["eps"] == eps[0] and d["degree"] == degree
                ]
            ),
            "-",
            color=colors[i],
            label=r"$\mathbf{\Phi}^{\text{X}} = \mathbf{\Phi}^{\text{CSR}}$; $p_{\hat{x}}=p_{\hat{y}}="
            + f"{degree}$",
        )
        plt.plot(
            [
                4 * (d["factor"] + d["degree"]) ** 2
                for d in data
                if d["eps"] == eps[0] and d["degree"] == degree and d["converged"][0]
            ],
            np.array(
                [
                    d["l2 error"][0][0]
                    for d in data
                    if d["eps"] == eps[0]
                    and d["degree"] == degree
                    and d["converged"][0]
                ]
            ),
            "o",
            color=colors[i],
        )
        plt.plot(
            [
                4 * (d["factor"] + d["degree"]) ** 2
                for d in data
                if d["eps"] == eps[0]
                and d["degree"] == degree
                and not d["converged"][0]
            ],
            np.array(
                [
                    d["l2 error"][0][0]
                    for d in data
                    if d["eps"] == eps[0]
                    and d["degree"] == degree
                    and not d["converged"][0]
                ]
            ),
            "o",
            color=colors[i],
            markerfacecolor="none",
        )
        print(
            [
                d["l2 error"][0][0]
                for d in data
                if d["eps"] == eps[0] and d["degree"] == degree
            ][-1]
        )

    plt.xlabel(r"$N_e \times N_{\hat{x}} \times N_{\hat{y}}$")
    plt.ylabel(
        r"$\epsilon_2\left(\mathbf{\Phi}^{\text{X}}, \mathbf{\Phi}^{\text{MC}}\right)$"
    )
    plt.xscale("log")
    plt.yscale("log")
    plt.legend(fontsize=10)
    plt.tight_layout()
    plt.savefig("./meshsize/figs/flux_l2error.eps", dpi=700, transparent=False)

    data = get_jsonl_data(
        dir / "meshsize/processed_meshsize.jsonl",
        lambda line_data: (
            (True, line_data["flux_stats"])
            if line_data["device"] == "gpu"
            else (False, None)
        ),
    )

    for case in ["Mixed", "Mixed (rounded)"]:
        for degree in degrees:
            plt.clf()
            for i in range(len(eps)):
                tt = [
                    np.array(
                        [
                            4 * (d["factor"] + d["degree"]) ** 2
                            for d in data
                            if case in d["solve_method"]
                            and d["eps"] == eps[i]
                            and d["degree"] == degree
                        ]
                    ),
                    np.array(
                        [
                            d["l2 error to csr"][
                                np.argwhere(
                                    case == np.array(d["solve_method"])
                                ).flatten()[0]
                            ][0]
                            for d in data
                            if case in d["solve_method"]
                            and d["eps"] == eps[i]
                            and d["degree"] == degree
                        ]
                    ),
                ]
                plt.plot(
                    tt[0],
                    tt[1],
                    linestyles[i] + "o",
                    color=colors[i],
                    label=rf"$\epsilon = {eps2str(eps[i])}$",
                )

            plt.xlabel(r"$N_e \times N_{\hat{x}} \times N_{\hat{y}}$")
            plt.ylabel(
                r"$\epsilon_2\left(\mathbf{\Phi}^{\text{"
                + case
                + r"}}; \mathbf{\Phi}^{\text{CSR}}\right)$"
            )
            plt.xscale("log")
            plt.yscale("log")
            plt.legend(fontsize=14)
            plt.tight_layout()
            plt.savefig(
                f"./meshsize/figs/flux_l2error2csr_p{degree}_{case}.eps",
                dpi=700,
                transparent=False,
            )
