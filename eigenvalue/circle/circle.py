import multiprocessing
import pickle

if __name__ == "__main__":
    multiprocessing.set_start_method("spawn")

import matplotlib.gridspec as gridspec
import matplotlib.pyplot as plt
import numpy as np
import torch as tn
from ttnte.assemblers import MatrixAssembler, TTAssembler
from ttnte.cad import Patch
from ttnte.cad.surfaces import circle
from ttnte.iga import IGAMesh
from ttnte.linalg import LinearSolverOptions, TTOperator, cpp_available, power
from ttnte.xs.benchmarks import pu239


def get_mesh(factor=5, degree=2):
    rc = 4.279960  # Critical radius (cm)

    # Create mesh
    mesh = IGAMesh()
    mesh.add_patch(Patch(circle(rc), "Pu-239"))

    # Refine mesh resolution
    mesh.refine(factor=factor, degree=degree)

    # Connect patches
    mesh.connect()

    # Finalize mesh
    mesh.finalize()
    return mesh


def evaluate_boundary(mesh):
    rc = 4.279960  # cm

    # Get flux along boundary
    center_flux = mesh((0.5, 0.5))[0][-1]
    points = np.zeros((400, 2))
    points[:100, 0] = np.linspace(0, 1, 100)
    points[100:200, 0] = 1
    points[100:200, 1] = np.linspace(0, 1, 100)
    points[200:300, 0] = np.linspace(0, 1, 100)[::-1]
    points[200:300, 1] = 1
    points[300:400, 1] = np.linspace(0, 1, 100)[::-1]

    points = mesh(points)[0]
    points[:, -1] /= center_flux

    # Convert to angle
    angular_points = np.zeros((points.shape[0], 2))
    angular_points[:, -1] = points[:, -1]
    angular_points[(points[:, 0] >= 0), 0] = np.arcsin(
        points[(points[:, 0] >= 0), 1] / rc
    )
    angular_points[(points[:, 0] < 0) & (points[:, 1] >= 0), 0] = (
        -np.arcsin(points[(points[:, 0] < 0) & (points[:, 1] >= 0), 1] / rc) + np.pi
    )
    angular_points[(points[:, 0] < 0) & (points[:, 1] < 0), 0] = (
        -np.arcsin(points[(points[:, 0] < 0) & (points[:, 1] < 0), 1] / rc) - np.pi
    )
    return angular_points[angular_points[:, 0].argsort()]


def evaluate_radius(mesh, radius, tol):
    # Plot and evaluate boundary flux
    center_flux = mesh((0.5, 0.5))[0][-1]

    # Calculate physical locations
    points = radius * np.ones((400, 2))
    angular_points = np.zeros((400, 2))
    angular_points[:, 0] = np.linspace(-np.pi, np.pi, 400)
    points[:, 0] *= np.cos(angular_points[:, 0])
    points[:, 1] *= np.sin(angular_points[:, 0])

    # Inverse map to the parametric domain
    points = mesh(mesh.inverse_map(points, tol=tol)[-1])[0][:, -1]
    angular_points[:, 1] = points / center_flux

    return angular_points


if __name__ == "__main__":
    if cpp_available == False:
        raise RuntimeError("C++ backend was not configured")

    # Make sure torch is using double precision by default
    tn.set_default_dtype(tn.float64)

    # Change number of threads used by PyTorch
    num_threads = 128 - 8
    tn.set_num_threads(num_threads)
    tn.set_num_interop_threads(num_threads)

    # Discretization
    num_ordinates = 4096
    factor = 10
    degree = 4

    # Critical radius
    rc = 4.279960  # cm

    # Get XS data
    xs_server = pu239(num_groups=1)

    # =====================================================================
    # Plot the problem
    # =====================================================================
    mesh = get_mesh(factor=factor, degree=degree)
    print(mesh)

    # =====================================================================
    # Assembler operators
    # =====================================================================
    # Create operators in COO format
    assembler = MatrixAssembler(
        mesh=mesh,
        xs_server=xs_server,
        num_ordinates=num_ordinates,
    )
    mats = assembler.build()

    # Save COO information
    assembler.save_info("./coo_info.csv")

    # Create operators in TT format
    assembler = TTAssembler(
        mesh=mesh,
        xs_server=xs_server,
        num_ordinates=num_ordinates,
        max_processes=4,
    )
    tts = assembler.build(use_tt=False, eps=1e-5)

    # Save TT information
    assembler.save_info("./tt_info.csv")

    # =====================================================================
    # Solve the problem (CSR only)
    # =====================================================================
    def csr_only():
        return (mats.H.clone() + mats.B_out.clone() - mats.S.clone()).combine(), mats.F

    def mixed():
        return (tts.H.clone() - tts.S.clone()).round(1e-5) + mats.B_out.clone(), tts.F

    def tts_only_rounded():
        return (tts.H.clone() - tts.S.clone() + tts.B_out.clone()).round(1e-5), tts.F

    # =====================================================================
    # Solve each problem
    # =====================================================================
    solutions = {}
    for name, get_ops in zip(
        ["CSR", "Mixed", "TT"],
        [csr_only, mixed, tts_only_rounded],
    ):
        T, F = get_ops()
        psi, k = power(
            T=T,
            F=F,
            tol=1e-8,
            maxiter=500,
            gpu_idx=0,
            lsoptions=LinearSolverOptions(restart=30, maxiter=10, tol=1e-10),
        )

        # Ravel solution back
        psi = psi.reshape(assembler.discretization)

        # Append solution
        solutions[name] = (psi, k)

    assert 0 == 1

    # Save solutions
    with open("solutions.pkl", "wb") as f:
        pickle.dump(solutions, f)

    # =====================================================================
    # Calculate statistics
    # =====================================================================
    stats = {
        "methods": ["CSR", "Mixed", "TT", "TT (rounded)"],
        "k": {
            "error": [],
        },
        "psi": {
            "0.5rc": {
                "l2 error": [],
            },
            "rc": {
                "l2 error": [],
            },
        },
    }
    for name, solution in solutions.items():
        # Get k and psi
        psi, k = solution

        # Integrate angular component to get scalar flux
        phi = assembler.angular_integral(psi).numpy()
        phi /= np.linalg.norm(phi.flatten())

        # Set scalar flux solution
        mesh.set_phi(phi[0,])

        # Eigenvalue statistics
        stats["k"]["error"].append((k - 1) * 1e5)

        # Flux statistics at r = 0.5rc
        points = evaluate_radius(mesh, 0.5 * rc, tol=1e-10)
        stats["psi"]["0.5rc"]["l2 error"].append(
            np.trapz((points[:, 1] - 0.8093) ** 2, points[:, 0])
            / (2 * np.pi * 0.8093**2)
        )

        # Flux statistics at r = rc
        points = evaluate_boundary(mesh)
        stats["psi"]["rc"]["l2 error"].append(
            np.trapz((points[:, 1] - 0.2926) ** 2, points[:, 0])
            / (2 * np.pi * 0.2926**2)
        )

    # Save results
    with open("stats.pkl", "wb") as f:
        pickle.dump(stats, f)

    # =====================================================================
    # Plot results
    # =====================================================================

    # Plot flux relative error at boundaries
    plt.clf()
    ax = plt.gca()
    for name, solution in solutions.items():
        psi, k = solution

        # Integrate angular component to get scalar flux
        phi = assembler.angular_integral(psi).numpy()
        phi /= np.linalg.norm(phi.flatten())

        # Set control points
        mesh.set_phi(phi[0,])

        points = evaluate_radius(mesh, 0.5 * rc, tol=1e-10)
        ax.plot(points[:, 0], np.abs(points[:, 1] - 0.8093) / 0.8093, label=name)

    ticks = [-np.pi, -0.5 * np.pi, 0, 0.5 * np.pi, np.pi]
    tick_labels = ["0", r"$\frac{\pi}{2}$", r"$\pi$", r"$\frac{3\pi}{2}$", r"$2\pi$"]

    ax.set_xticks(ticks)
    ax.set_xticklabels(tick_labels)
    ax.set_xlabel(r"$\theta$")
    ax.set_ylabel(r"$\epsilon_1(r = 0.5r_c, \theta)$")
    plt.legend()
    plt.tight_layout()
    plt.savefig(f"./figs/error_0.5rc.png", dpi=300)

    plt.clf()
    ax = plt.gca()
    for name, solution in solutions.items():
        psi, k = solution

        # Integrate angular component to get scalar flux
        phi = assembler.angular_integral(psi).numpy()
        phi /= np.linalg.norm(phi.flatten())

        # Set control points
        mesh.set_phi(phi[0,])

        points = evaluate_boundary(mesh)
        ax.plot(points[:, 0], np.abs(points[:, 1] - 0.2926) / 0.2926, label=name)

    ticks = [-np.pi, -0.5 * np.pi, 0, 0.5 * np.pi, np.pi]
    tick_labels = ["0", r"$\frac{\pi}{2}$", r"$\pi$", r"$\frac{3\pi}{2}$", r"$2\pi$"]

    ax.set_xticks(ticks)
    ax.set_xticklabels(tick_labels)
    ax.set_xlabel(r"$\theta$")
    ax.set_ylabel(r"$\epsilon_1(r = r_c, \theta)$")
    plt.legend()
    plt.tight_layout()
    plt.savefig(f"./figs/error_rc.png", dpi=300)

    psi, k = solutions[stats["methods"][np.argmin(stats["k"]["error"])]]

    # Integrate angular component to get scalar flux
    phi = assembler.angular_integral(psi).numpy()
    phi /= np.linalg.norm(phi.flatten())

    # Set control points
    mesh.set_phi(phi[0,])

    # Plot
    plt.clf()
    ax = mesh.plot(plot_ctrlpts=False, use_3d=True, figsize=(8, 6))
    fig = ax.figure
    # Get the position of the main 3D axes
    pos = ax.get_position()  # returns Bbox: (x0, y0, x1, y1)

    # Choose a new width for the colorbar (e.g., 40% of figure)
    cbar_width = 0.03
    cbar_height = 0.4
    cbar_bottom = 0.18  # your chosen vertical position

    # Compute centered left coordinate relative to the 3D plot
    left = pos.x0 + (pos.width - cbar_width) / 2 + 0.005
    cax = fig.add_axes([left + 0.19, cbar_bottom + 0.085, cbar_width, cbar_height])
    cbar = fig.colorbar(ax.collections[0], cax=cax, orientation="vertical")
    cbar.ax.tick_params(labelrotation=0, labelsize=12)
    cbar.set_label("$\\phi(\\hat{x}, \\hat{y})$", rotation=0, labelpad=25, fontsize=14)
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
    plt.savefig(f"./figs/phi.png", dpi=300)
