import multiprocessing
import pickle
import sys

if __name__ == "__main__":
    multiprocessing.set_start_method("spawn")
    sys.path.append("../..")

import matplotlib.pyplot as plt
import numpy as np
import torch as tn
from matplotlib.patches import Polygon
from mpl_toolkits.axes_grid1 import make_axes_locatable
from pincell import get_mesh, get_xs
from ttnte.assemblers import MatrixAssembler

# Change plotting label sizes
plt.rcParams["font.size"] = 14
plt.rcParams["axes.labelsize"] = 16
plt.rcParams["xtick.labelsize"] = 14
plt.rcParams["ytick.labelsize"] = 14
plt.rcParams["legend.fontsize"] = 14
plt.rcParams["axes.grid"] = True

if __name__ == "__main__":
    # Solutions from OpenMC
    k_mc = [1.325593334007463, 3.229591167123522e-05]
    phi_mc = np.load(
        "/home/myerspat/research/tensor_trains/tt_nte/notebooks/eigenvalue/pincell/openmc/data/mesh_flux.npy"
    )
    phi_mc_stdev = np.load(
        "/home/myerspat/research/tensor_trains/tt_nte/notebooks/eigenvalue/pincell/openmc/data/mesh_stdev.npy"
    )
    phi_mc_stdev /= np.linalg.norm(phi_mc.flatten(), 2)
    phi_mc /= np.linalg.norm(phi_mc.flatten(), 2)

    # Read in data
    with open("data.pkl", "rb") as f:
        data = pickle.load(f)

    # ========================================================================
    # Get an example mesh
    # ========================================================================
    mesh = get_mesh(factor=data["factor"], degree=data["degree"])
    ax = mesh.plot(
        plot_ctrlpts=False,
        figsize=(6, 6),
        color_by="material",
        colors={"UO2": "#E69F00", "Water": "#0072B2"},
    )

    # Plot boundaries of patches
    for patch in mesh.patches.values():
        # Get boundary in parametric space
        coords = np.zeros((4, 128, 2))
        coords[0, :, 0] = np.linspace(0, 1, 128)
        coords[1, :, 0] = 1
        coords[1, :, 1] = np.linspace(0, 1, 128)
        coords[2, :, 1] = 1
        coords[2, :, 0] = np.linspace(0, 1, 128)[::-1]
        coords[3, :, 1] = np.linspace(0, 1, 128)[::-1]

        boundary = patch(coords.reshape((-1, 2)))

        # Create outline
        outline = Polygon(
            boundary[:, :-1],
            closed=True,
            edgecolor="black",
            facecolor="none",
            linewidth=1.5,
        )
        ax.add_patch(outline)

    plt.tight_layout()
    dx = 0.01 * (ax.get_xlim()[1] - ax.get_xlim()[0])
    dy = 0.01 * (ax.get_ylim()[1] - ax.get_ylim()[0])
    plt.xlim((ax.get_xlim()[0] - dx, ax.get_xlim()[1] + dx))
    plt.ylim((ax.get_ylim()[0] - dy, ax.get_ylim()[1] + dy))
    plt.grid(False)
    plt.savefig("./figs/pincell.png", dpi=300, transparent=True)

    # Create matrix assembler
    assembler = MatrixAssembler(mesh, get_xs(data["num_groups"]), data["num_ordinates"])

    # Calculate scalar flux
    phi = assembler.angular_integral(
        tn.tensor(
            data["psi"]["value"][
                np.argwhere(np.array(data["solve_method"]) == "CSR").flatten()[0]
            ]
        ).reshape(assembler.discretization)
    )

    # Plot CSR
    for g in range(data["num_groups"]):
        plt.clf()
        mesh.set_phi(phi[0,])
        ax, cbar = mesh.plot(plot_ctrlpts=False)
        cbar.set_label(r"$\phi_{" + str(g + 1) + r"}(\hat{x}, \hat{y})$")
        plt.grid(False)
        plt.tight_layout()
        plt.savefig(f"./figs/phi_{g + 1}.png", dpi=300, transparent=True)

    # Get mapping for mesh element averaging
    pids, coords = mesh.map_regular_mesh(shape=phi_mc.shape[1:], N=(5, 5))

    # =====================================================================
    # Calculate statistics
    # =====================================================================
    X, Y = np.meshgrid(
        np.linspace(0, 5, phi_mc.shape[1]),
        np.linspace(0, 5, phi_mc.shape[2]),
    )
    data["k"]["error"] = []
    data["k"]["relerror"] = []
    data["k"]["zscore"] = []
    for error_name in ["error", "relative_error", "zscore"]:
        data["psi"][error_name] = {
            "minimum": [],
            "q1": [],
            "median": [],
            "q2": [],
            "maximum": [],
            "mean": [],
        }
    data["psi"]["l2 error"] = []

    i = 0
    for name, psi in zip(data["solve_method"], data["psi"]["value"]):
        # Calculate k errors
        data["k"]["error"].append(data["k"]["value"][i] - k_mc[0])
        data["k"]["relerror"].append(np.abs(data["k"]["error"][-1]) / k_mc[0])
        data["k"]["zscore"].append(np.abs(data["k"]["error"][-1]) / k_mc[1])

        # Compute scalar flux
        phi = assembler.angular_integral(
            tn.tensor(psi).reshape(assembler.discretization)
        ).numpy()

        error_methods = {
            "error": lambda a, b, c: a - b,
            "relative_error": lambda a, b, c: np.abs(a - b) / b,
            "zscore": lambda a, b, c: np.abs(a - b) / c,
        }
        plot_labals = {
            "error": lambda case, g: r"$\mathbf{\Phi}_"
            + str(g)
            + "^{"
            + case
            + r"}-\mathbf{\Phi}^{MC}_"
            + str(g)
            + "$",
            "relative_error": lambda case, g: r"$\frac{\left|\mathbf{\Phi}_"
            + str(g)
            + "^{"
            + case
            + r"}-\mathbf{\Phi}^{MC}_"
            + str(g)
            + r"\right|}{\mathbf{\Phi}^{MC}_"
            + str(g)
            + "}$",
            "zscore": lambda case, g: r"$\mathbf{z}_" + str(g) + "^{" + case + "}$",
        }

        # Calculate average scalar flux
        phi_avg = np.zeros(phi_mc.shape)

        # Calculate phi_avg and errors
        for g in range(data["num_groups"]):
            # Set control points
            mesh.set_phi(phi[g,])

            # Calculate regular mesh
            phi_avg[g,] = mesh.regular_mesh(pids, coords)

        # Normalize
        phi_avg /= np.linalg.norm(phi_avg.flatten(), 2)

        for error_name, error_method in error_methods.items():
            # Calculate error
            error = error_method(phi_avg, phi_mc, phi_mc_stdev)

            # Calculate flux statistics
            minimum = np.zeros((data["num_groups"] + 1))
            q1 = np.zeros((data["num_groups"] + 1))
            median = np.zeros((data["num_groups"] + 1))
            q2 = np.zeros((data["num_groups"] + 1))
            maximum = np.zeros((data["num_groups"] + 1))
            mean = np.zeros((data["num_groups"] + 1))

            for g in range(data["num_groups"]):
                minimum[g] = np.min(error[g,])
                q1[g] = np.percentile(error[g,], 25)
                median[g] = np.median(error[g,])
                q2[g] = np.percentile(error[g,], 75)
                maximum[g] = np.max(error[g,])
                mean[g] = np.mean(error[g,])

                # Plot errors
                plt.clf()
                ax = plt.gca()
                cmesh = ax.pcolormesh(
                    X, Y, error[g,].reshape(phi_mc.shape[1:]), cmap="plasma"
                )
                divider = make_axes_locatable(ax)
                cbar = plt.colorbar(
                    cmesh,
                    cax=divider.append_axes("right", size="5%", pad=0.05),
                )
                cbar.set_label(plot_labals[error_name](name, g))
                ax.set_aspect("equal")
                ax.set_xlabel(r"$x(\hat{x}, \hat{y})~(cm)$")
                ax.set_ylabel(r"$y(\hat{x}, \hat{y})~(cm)$")
                ax.axis("off")
                plt.tight_layout()
                plt.savefig(
                    f"./figs/{error_name}_{g + 1}_{name}.png", transparent=True, dpi=300
                )

            minimum[-1] = np.min(minimum[:-1])
            q1[-1] = np.percentile(error, 25)
            median[-1] = np.median(error)
            q2[-1] = np.percentile(error, 75)
            maximum[-1] = np.max(maximum[:-1])
            mean[-1] = np.mean(error)

            # Add this to results
            data["psi"][error_name]["minimum"].append(minimum)
            data["psi"][error_name]["q1"].append(q1)
            data["psi"][error_name]["median"].append(median)
            data["psi"][error_name]["q2"].append(q2)
            data["psi"][error_name]["maximum"].append(maximum)
            data["psi"][error_name]["mean"].append(mean)

        # Calculate l2 error
        l2 = np.zeros((data["num_groups"] + 1))
        for g in range(data["num_groups"]):
            l2[g] = np.linalg.norm(phi_avg[g,] - phi_mc[g,], 2) / np.linalg.norm(
                phi_mc[g,], 2
            )
        l2[-1] = np.sqrt(np.sum(l2[:-1] ** 2))

        # Add this to the results
        data["psi"]["l2 error"].append(l2)
        i += 1

    # Save data
    with open("data.pkl", "wb") as f:
        pickle.dump(data, f)
