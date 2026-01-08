import multiprocessing
import os
import pickle
import sys
from pathlib import Path

if __name__ == "__main__":
    multiprocessing.set_start_method("forkserver")
    sys.path.append("../..")

import matplotlib.pyplot as plt
import numpy as np
import torch as tn
from circle import get_mesh, get_xs
from scipy import stats
from ttnte.assemblers import MatrixAssembler

from extract import get_jsonl_data, get_pickle_data

# Change plotting label sizes
plt.rcParams["font.size"] = 14
# plt.rcParams["axes.titlsize"] = 14
plt.rcParams["axes.labelsize"] = 16
plt.rcParams["xtick.labelsize"] = 14
plt.rcParams["ytick.labelsize"] = 14
plt.rcParams["legend.fontsize"] = 10
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
    (dir / "direction/figs").mkdir(parents=True, exist_ok=True)

    # Solutions from OpenMC
    leakage_frac_openmc = [0.43995423399999983, 2.2245143201699137e-05]

    num_ordinates = [16, 64, 256, 1024, 4096, 16384, 65536, 262144]
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
    # Get an example mesh
    # ========================================================================
    factor = 10
    degree = 2
    N = 16384
    mesh = get_mesh(factor=10, degree=2)
    mesh.plot(figsize=(6, 6))
    plt.savefig("./direction/figs/circle.png", dpi=300, transparent=True)

    # Create matrix assembler
    assembler = MatrixAssembler(mesh, get_xs(1), 16384)

    # Load data
    psi = pickle.load(
        open(
            f"./direction/meshes/N{N}_G1_A{factor + degree}_B{factor + degree}_p{degree}_q{degree}_eps1e-08cpu.pkl",
            "rb",
        )
    )["CSR"].reshape(assembler.discretization)

    # Calculate scalar flux
    phi = assembler.angular_integral(tn.tensor(psi))

    # Plot data
    plt.clf()
    mesh.set_phi(phi[0,])
    ax = mesh.plot(plot_ctrlpts=False, use_3d=True, figsize=(6, 8))
    fig = ax.figure
    # Get the position of the main 3D axes
    pos = ax.get_position()  # returns Bbox: (x0, y0, x1, y1)

    # Choose a new width for the colorbar (e.g., 40% of figure)
    cbar_width = 0.5
    cbar_height = 0.03
    cbar_bottom = 0.18  # your chosen vertical position

    # Compute centered left coordinate relative to the 3D plot
    left = pos.x0 + (pos.width - cbar_width) / 2 + 0.005
    cax = fig.add_axes([left, cbar_bottom + 0.05, cbar_width, cbar_height])
    cbar = fig.colorbar(ax.collections[0], cax=cax, orientation="horizontal")
    cbar.ax.tick_params(labelrotation=0, labelsize=12)
    cbar.set_label("$\\phi(\\hat{x}, \\hat{y})$", rotation=0, fontsize=14)
    ax.grid(False)
    # Turn off panes and ticks
    ax.set_xticks([])
    ax.set_yticks([])
    ax.set_zticks([])
    ax.set_ylabel(None)
    ax.set_xlabel(None)

    ax.xaxis.pane.set_visible(False)
    ax.yaxis.pane.set_visible(False)
    ax.zaxis.pane.set_visible(False)

    # Turn off axis lines
    ax.xaxis.line.set_color((1.0, 1.0, 1.0, 0.0))  # Invisible line
    ax.yaxis.line.set_color((1.0, 1.0, 1.0, 0.0))
    ax.zaxis.line.set_color((1.0, 1.0, 1.0, 0.0))
    plt.savefig(f"./direction/figs/phi.png", dpi=300, transparent=True)

    # ========================================================================
    # Leakage Fraction Plots
    # ========================================================================

    # Extract leakage fraction data
    data = get_jsonl_data(
        dir / "direction/processed_direction.jsonl",
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
        num_ordinates[0],
        num_ordinates[-1],
        label=r"$f_{\text{Leak}}^{\text{ref}}\pm 2\sigma$",
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
            label=r"CSR: $p_{\hat{x}} = p_{\hat{y}} = " + f"{degree}$",
        )
    plt.ylabel(r"$f_{\text{Leak}}$")
    plt.xlabel(r"$N_{\Omega}$")
    plt.xscale("log")
    plt.grid()
    plt.legend()
    plt.tight_layout()
    plt.savefig("./direction/figs/leakage.png", dpi=300, transparent=True)

    # Plot CSR leakage fraction Z-score to OpenMC
    plt.clf()
    plt.hlines(
        [2],
        num_ordinates[0],
        num_ordinates[-1],
        label=r"$2\sigma$",
        color="black",
    )
    plt.hlines(
        [1],
        num_ordinates[0],
        num_ordinates[-1],
        linestyles="--",
        label=r"$\sigma$",
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
                if d["eps"] == eps[0]
                and d["degree"] == degree
                and "CSR" in d["solve_method"]
            ],
            [
                abs(
                    d["zscore"][
                        np.argwhere(np.array(d["solve_method"]) == "CSR").flatten()[0]
                    ]
                )
                for d in data
                if d["eps"] == eps[0]
                and d["degree"] == degree
                and "CSR" in d["solve_method"]
            ],
            "-o",
            label=r"CSR: $p_{\hat{x}} = p_{\hat{y}} = " + f"{degree}$",
        )
    plt.ylabel(r"Number of $\sigma$ from $f_{\text{Leak}}^{\text{ref}}$")
    plt.xlabel(r"$N_{\Omega}$")
    plt.xscale("log")
    plt.yscale("log")
    plt.legend(ncol=2)
    plt.tight_layout()
    plt.savefig("./direction/figs/leakage_zscore.png", dpi=300, transparent=True)

    # Plot CSR leakage fraction error to OpenMC
    plt.clf()
    plt.hlines(
        [(2 * leakage_frac_openmc[1]) / leakage_frac_openmc[0]],
        num_ordinates[0],
        num_ordinates[-1],
        label=r"$f_{\text{leak}}^{\text{X}} = f_{\text{leak}}^{\text{MC}} + 2\sigma$",
        color="black",
    )
    plt.hlines(
        [(leakage_frac_openmc[1]) / leakage_frac_openmc[0]],
        num_ordinates[0],
        num_ordinates[-1],
        linestyles="--",
        label=r"$f_{\text{leak}}^{\text{X}} = f_{\text{leak}}^{\text{MC}} + \sigma$",
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
            label=r"$f_{\text{leak}}^{\text{X}} = f_{\text{leak}}^{\text{CSR}}$; $p_{\hat{x}} = p_{\hat{y}} = "
            + f"{degree}$",
        )
    plt.ylabel(
        r"$\delta f\left(f_{\text{leak}}^{\text{X}}, f_{\text{leak}}^{\text{MC}}\right)$"
    )
    plt.xlabel(r"$N_\Omega$")
    plt.xscale("log")
    plt.yscale("log")
    plt.legend(ncol=2)
    plt.tight_layout()
    plt.savefig("./direction/figs/leakage_relerror.png", dpi=300, transparent=True)

    # Look at errors relative to CSR
    plt.clf()
    for degree in degrees:
        # Get CSR solution
        csr = [
            np.array(
                [
                    d["num_ordinates"]
                    for d in data
                    if d["eps"] == eps[0]
                    and d["degree"] == degree
                    and "CSR" in d["solve_method"]
                ]
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
        for solve_method in ["TT", "TT (rounded)", "Mixed", "Mixed (rounded)"]:
            plt.clf()
            for i in range(len(eps)):
                method = np.array(
                    [
                        np.array(
                            [
                                d["num_ordinates"]
                                for d in data
                                if d["eps"] == eps[i]
                                and d["degree"] == degree
                                and solve_method in d["solve_method"]
                            ]
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

            plt.xlabel(r"$N_\Omega$")
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
                f"./direction/figs/leakage_relerror_p{degree}_{solve_method}.png",
                dpi=300,
                transparent=True,
            )

    # ========================================================================
    # Ranks and Compression
    # ========================================================================

    # Get ranks for all operators
    data = get_jsonl_data(
        dir / "direction/processed_direction.jsonl",
        lambda line_data: (
            (True, line_data["ranks"])
            if line_data["device"] == "gpu"
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
                    label=rf"$\epsilon={eps2str(eps[i])}$",
                )
            plt.xlabel(r"$N_{\Omega}$")
            plt.ylabel(r"$r_{\text{max}}\left(" + prettyOp(op, "TT") + r"\right)$")
            plt.xscale("log")
            plt.legend()
            plt.tight_layout()
            plt.savefig(
                f"./direction/figs/ranks_p{degree}_{op}.png", dpi=300, transparent=True
            )

    data = get_jsonl_data(
        dir / "direction/processed_direction.jsonl",
        lambda line_data: (
            (True, line_data["compression"]["tts"])
            if line_data["device"] == "cpu"
            else (False, None)
        ),
    )
    data2 = get_jsonl_data(
        dir / "direction/processed_direction.jsonl",
        lambda line_data: (
            (True, line_data["compression"]["matrix"])
            if line_data["device"] == "cpu" and line_data["eps"] == eps[0]
            else (False, None)
        ),
    )

    plt.clf()
    for op in ["H", "S", "B_out"]:
        print(op)
        for degree in degrees:
            print(f"degree = {degree}")
            plt.clf()
            plt.plot(
                [d["num_ordinates"] for d in data2 if d["degree"] == degree],
                [np.array(d[op]).max() for d in data2 if d["degree"] == degree],
                "--o",
                label=r"CSR",
            )
            csr_data = [
                np.array([d["num_ordinates"] for d in data2 if d["degree"] == degree])[
                    :3
                ],
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
                np.array([d["num_ordinates"] for d in data2 if d["degree"] == degree])[
                    -3:
                ],
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
                            d["num_ordinates"]
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
                            d["num_ordinates"]
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
                    label=rf"TT, $\epsilon={eps2str(eps[i])}$",
                )
            print()
            plt.xlabel(r"$N_{\Omega}$")
            plt.ylabel(r"$\text{CR}\left(" + prettyOp(op) + r"\right)$")
            plt.xscale("log")
            plt.yscale("log")
            plt.legend()
            plt.tight_layout()
            plt.savefig(
                f"./direction/figs/compression_p{degree}_{op}.png",
                dpi=300,
                transparent=True,
            )

            plt.clf()
            csr = [
                np.array([d["num_ordinates"] for d in data2 if d["degree"] == degree]),
                np.array(
                    [np.array(d[op]).max() for d in data2 if d["degree"] == degree]
                ),
            ]
            for i in range(len(eps)):
                tt = [
                    np.array(
                        [
                            d["num_ordinates"]
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
            plt.xlabel(r"$N_{\Omega}$")
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
                f"./direction/figs/compression_ratio_p{degree}_{op}.png",
                dpi=300,
                transparent=True,
            )

    data = get_jsonl_data(
        dir / "direction/processed_direction.jsonl",
        lambda line_data: (
            (True, {"total": line_data["compression"]["total"]})
            if line_data["device"] == "cpu"
            else (False, None)
        ),
    )

    for degree in degrees:
        for i in range(len(eps)):
            plt.clf()
            plt.plot(
                [
                    d["num_ordinates"]
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
                # label=f"{name}: "
                # + r"$p_{\hat{x}} = p_{\hat{y}} = "
                # + f"{degree}"
                # + ("" if name == "CSR" else rf", \epsilon={eps2str(eps[i])}")
                # + "$",
            )

            for j, name in enumerate(
                ["TT", "Mixed", "TT (rounded)", "Mixed (rounded)"]
            ):
                plt.plot(
                    [
                        d["num_ordinates"]
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
                    # label=f"{name}: "
                    # + r"$p_{\hat{x}} = p_{\hat{y}} = "
                    # + f"{degree}"
                    # + ("" if name == "CSR" else rf", \epsilon={eps2str(eps[i])}")
                    # + "$",
                )

            plt.xlabel(r"$N_{\Omega}$")
            plt.ylabel(
                r"$\text{CR}\left("
                + prettyOp("T")
                + r"\right)$; $p_{\hat{x}} = p_{\hat{y}} = "
                + rf"{degree}$, $\epsilon={eps2str(eps[i])}$"
            )
            plt.xscale("log")
            plt.yscale("log")
            plt.legend()
            plt.tight_layout()
            plt.savefig(
                f"./direction/figs/compression_p{degree}_eps{eps[i]}_T.png",
                dpi=300,
                transparent=True,
            )

    # Get angular flux compression ranks and compression
    data = get_jsonl_data(
        dir / "direction/processed_direction.jsonl",
        lambda line_data: (
            (
                True,
                {
                    "ranks": line_data["flux_stats"]["ranks"],
                    "compression": line_data["flux_stats"]["compression"],
                },
            )
            if line_data["device"] == "cpu"
            else (False, None)
        ),
    )

    for degree in degrees:
        plt.clf()
        for i in range(len(eps)):
            if i == 0:
                num_ords = np.array(
                    [
                        d["num_ordinates"]
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
                    d["num_ordinates"]
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

        plt.xlabel(r"$N_\Omega$")
        plt.ylabel(r"$r_{\text{max}}\left(\mathbf{\Psi}\right)$")
        plt.xscale("log")
        plt.yscale("log")
        plt.legend(fontsize=14)
        plt.tight_layout()
        plt.savefig(
            f"./direction/figs/ranks_p{degree}_psi.png", dpi=300, transparent=True
        )

    for degree in degrees:
        plt.clf()
        for i in range(len(eps)):
            if i == 0:
                num_ords = np.array(
                    [
                        d["num_ordinates"]
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
                    d["num_ordinates"]
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

        plt.xlabel(r"$N_\Omega$")
        plt.ylabel(r"$\text{CR}\left(\mathbf{\Psi}^{\text{TT}}\right)$")
        plt.xscale("log")
        plt.yscale("log")
        plt.legend(fontsize=14)
        plt.tight_layout()
        plt.savefig(
            f"./direction/figs/compression_p{degree}_psi.png", dpi=300, transparent=True
        )

    # ========================================================================
    # Matvec scaling
    # ========================================================================

    # Get Matvec information, In this case we don't care about eps as it had no effect
    data = get_jsonl_data(
        dir / "direction/processed_direction.jsonl",
        lambda line_data: (
            (True, line_data["matvec"]) if line_data["eps"] == eps[0] else (False, None)
        ),
    )

    for degree in degrees:
        plt.clf()
        for device in ["cpu", "gpu"]:
            for i, case in enumerate(
                ["CSR", "TT", "Mixed", "TT (rounded)", "Mixed (rounded)"]
            ):
                num_ords = np.array(
                    [
                        d["num_ordinates"]
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
                plt.plot(
                    num_ords[
                        np.isin(
                            num_ords,
                            np.array(num_ordinates)[(2 if degree == 2 else 0) :],
                        )
                    ],
                    time[
                        np.isin(
                            num_ords,
                            np.array(num_ordinates)[(2 if degree == 2 else 0) :],
                        )
                    ],
                    ("-" if device == "cpu" else "--") + "o",
                    color=colors[i],
                    label=f"{case}, {device.upper()}",
                )
                plt.fill_between(
                    num_ords[
                        np.isin(
                            num_ords,
                            np.array(num_ordinates)[(2 if degree == 2 else 0) :],
                        )
                    ],
                    (time - stdev)[
                        np.isin(
                            num_ords,
                            np.array(num_ordinates)[(2 if degree == 2 else 0) :],
                        )
                    ],
                    (time + stdev)[
                        np.isin(
                            num_ords,
                            np.array(num_ordinates)[(2 if degree == 2 else 0) :],
                        )
                    ],
                    color=colors[i],
                    alpha=0.5,
                )

        plt.xlabel(r"$N_{\Omega}$")
        plt.ylabel(r"Average SpMV Time $(ms)$")
        plt.xscale("log")
        plt.yscale("log")
        plt.legend(ncol=2, fontsize=10)
        plt.tight_layout()
        plt.savefig(
            f"./direction/figs/matvec_time_p{degree}_eps{eps[0]}.png",
            dpi=300,
            transparent=True,
        )

    # ========================================================================
    # GMRES scaling
    # ========================================================================

    # Get GMRES information, In this case we don't care about eps as it had no effect
    data = get_jsonl_data(
        dir / "direction/processed_direction.jsonl",
        lambda line_data: (
            (True, line_data["gmres"]) if line_data["eps"] == eps[0] else (False, None)
        ),
    )

    for degree in degrees:
        plt.clf()
        for device in ["cpu", "gpu"]:
            for i, case in enumerate(
                ["CSR", "TT", "Mixed", "TT (rounded)", "Mixed (rounded)"]
            ):
                num_ords = np.array(
                    [
                        d["num_ordinates"]
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

        plt.xlabel(r"$N_{\Omega}$")
        plt.ylabel(r"GMRES Run Time $(s)$")
        plt.xscale("log")
        plt.yscale("log")
        plt.legend(ncol=2, fontsize=10)
        plt.tight_layout()
        plt.savefig(
            f"./direction/figs/gmres_time_p{degree}_eps{eps[0]}.png",
            dpi=300,
            transparent=True,
        )
