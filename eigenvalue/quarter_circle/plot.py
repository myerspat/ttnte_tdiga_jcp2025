import multiprocessing
import pickle
import sys

if __name__ == "__main__":
    multiprocessing.set_start_method("spawn")
    sys.path.append("../..")

import matplotlib.pyplot as plt
import numpy as np
import torch as tn
from quarter_circle import get_mesh, get_xs
from ttnte.assemblers import MatrixAssembler

# Change plotting label sizes
plt.rcParams["font.size"] = 14
plt.rcParams["axes.labelsize"] = 16
plt.rcParams["xtick.labelsize"] = 14
plt.rcParams["ytick.labelsize"] = 14
plt.rcParams["legend.fontsize"] = 14


def evaluate_boundary(mesh):
    rc = 4.279960

    # Plot and evaluate boundary flux
    center_flux = mesh((0, 0))[0][-1]
    points = np.ones((400, 2))
    points[:, 0] = np.linspace(0, 1, 400)

    points = mesh(points)[0]
    points[:, -1] /= center_flux

    # Convert to angle
    angular_points = np.zeros((points.shape[0], 2))
    angular_points[:, -1] = points[:, -1]
    angular_points[:, 0] = np.arcsin(points[(points[:, 0] >= 0), 1] / rc)
    return angular_points[angular_points[:, 0].argsort()]


def evaluate_radius(mesh, radius, tol):
    # Plot and evaluate boundary flux
    center_flux = mesh((0, 0))[0][-1]

    # Calculate physical locations
    points = radius * np.ones((400, 2))
    angular_points = np.zeros((400, 2))
    angular_points[:, 0] = np.linspace(0, np.pi / 2, 400)
    points[:, 0] *= np.cos(angular_points[:, 0])
    points[:, 1] *= np.sin(angular_points[:, 0])

    # Inverse map to the parametric domain
    points = mesh(mesh.inverse_map(points, tol=tol)[-1])[0][:, -1]
    angular_points[:, 1] = points / center_flux

    return angular_points


if __name__ == "__main__":
    rc = 4.279960

    # Read in data
    with open("data.pkl", "rb") as f:
        data = pickle.load(f)

    # ========================================================================
    # Get an example mesh
    # ========================================================================
    mesh = get_mesh(factor=data["factor"], degree=data["degree"])
    ax = mesh.plot(
        plot_ctrlpts=True,
        figsize=(6, 6),
        color_by="material",
    )
    plt.tight_layout()
    plt.savefig("./figs/quarter_circle.png", dpi=300, transparent=True)

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
    plt.clf()
    mesh.set_phi(phi[0,])
    ax, cbar = mesh.plot(plot_ctrlpts=False)
    cbar.set_label(r"$\phi(\hat{x}, \hat{y})$")
    plt.tight_layout()
    plt.savefig(f"./figs/phi.png", dpi=300, transparent=True)

    # =====================================================================
    # Calculate statistics
    # =====================================================================
    data["k"]["error"] = []
    data["psi"]["0.5rc l2 error"] = []
    data["psi"]["rc l2 error"] = []

    i = 0
    plt.clf()
    ax = plt.gca()
    for name, psi in zip(data["solve_method"], data["psi"]["value"]):
        # Calculate k errors
        data["k"]["error"].append(data["k"]["value"][i] - 1)

        # Compute scalar flux
        phi = assembler.angular_integral(
            tn.tensor(psi).reshape(assembler.discretization)
        ).numpy()
        phi /= np.linalg.norm(phi.flatten())
        mesh.set_phi(phi[0,])

        # Flux statistics at r = 0.5rc
        points = evaluate_radius(mesh, 0.5 * rc, tol=1e-10)
        data["psi"]["0.5rc l2 error"].append(
            np.trapz((points[:, 1] - 0.8093) ** 2, points[:, 0])
            / (np.pi / 2 * 0.8093**2)
        )

        # Plot error
        ax.plot(points[:, 0], np.abs(points[:, 1] - 0.8093) / 0.8093, label=name)
        i += 1

    ticks = [0, np.pi / 6, np.pi / 4, np.pi / 3, np.pi / 2]
    tick_labels = [
        "0",
        r"$\frac{\pi}{6}$",
        r"$\frac{\pi}{4}$",
        r"$\frac{\pi}{2}$",
        r"$\frac{\pi}{2}$",
    ]

    ax.set_xticks(ticks)
    ax.set_xticklabels(tick_labels)
    ax.set_xlabel(r"$\theta$")
    ax.set_ylabel(r"$\delta \phi(r = 0.5r_c, \theta)$")
    plt.legend()
    plt.grid(True)
    plt.tight_layout()
    plt.savefig(f"./figs/error_0.5rc.png", dpi=300)

    plt.clf()
    ax = plt.gca()
    for name, psi in zip(data["solve_method"], data["psi"]["value"]):
        # Compute scalar flux
        phi = assembler.angular_integral(
            tn.tensor(psi).reshape(assembler.discretization)
        ).numpy()
        phi /= np.linalg.norm(phi.flatten())
        mesh.set_phi(phi[0,])

        # Flux statistics at r = 0.5rc
        points = evaluate_boundary(mesh)
        data["psi"]["rc l2 error"].append(
            np.trapz((points[:, 1] - 0.2926) ** 2, points[:, 0])
            / (np.pi / 2 * 0.2926**2)
        )

        # Plot error
        ax.plot(points[:, 0], np.abs(points[:, 1] - 0.2926) / 0.2926, label=name)

    ticks = [0, np.pi / 6, np.pi / 4, np.pi / 3, np.pi / 2]
    tick_labels = [
        "0",
        r"$\frac{\pi}{6}$",
        r"$\frac{\pi}{4}$",
        r"$\frac{\pi}{2}$",
        r"$\frac{\pi}{2}$",
    ]

    ax.set_xticks(ticks)
    ax.set_xticklabels(tick_labels)
    ax.set_xlabel(r"$\theta$")
    ax.set_ylabel(r"$\delta \phi(r = r_c, \theta)$")
    plt.legend()
    plt.grid(True)
    plt.tight_layout()
    plt.savefig(f"./figs/error_rc.png", dpi=300)

    # Save data
    with open("data.pkl", "wb") as f:
        pickle.dump(data, f)
