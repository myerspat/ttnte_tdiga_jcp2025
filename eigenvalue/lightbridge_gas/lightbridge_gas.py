import multiprocessing
import pickle

if __name__ == "__main__":
    multiprocessing.set_start_method("spawn")

import matplotlib.pyplot as plt
import numpy as np
import torch as tn
from igakit import cad
from matplotlib.patches import Polygon
from mpl_toolkits.axes_grid1 import make_axes_locatable
from ttnte.assemblers import MatrixAssembler, TTAssembler
from ttnte.cad import Patch
from ttnte.cad.curves import qtrlobe
from ttnte.iga import IGAMesh
from ttnte.linalg import LinearSolverOptions, cpp_available, power
from ttnte.xs.benchmarks import kaist


def get_mesh(
    factor=5, degree=2, materials=["BA (UO2 FA)", "UO2 3%", "Guide Tube", "Water"]
):
    D = 1.26  # Fuel width
    D2 = D * 0.5
    X = 1.36  # Channel pitch
    delta = 0.306  # Width of lobes
    y2 = delta * 0.5
    d = 0.04  # Thickness of cladding at valleys
    dmax = 0.102  # Thickness of cladding at ends of the lobes
    R = 0.297  # Radius defining outer curve of valleys
    a = 0.156  # Displacer width

    y1 = y2 - d  # Half of width of inner lobe
    x1 = D2 - R - y2 - dmax  # Portrusion of innerlobe
    x2 = x1 + dmax  # Portrusion of outer lobe

    # NURBS curves
    origin = cad.line(p0=(0, 0), p1=(0, 0))
    burn = cad.line(p1=(a / (2**0.5), 0), p0=(0, a / (2**0.5)))
    fuel = qtrlobe(outrad=R + d, portrs=x1, hfwidth=y1)
    clad = qtrlobe(outrad=R, portrs=x2, hfwidth=y2)
    topedge = cad.line(p0=(0, X / 2), p1=(X / 2, X / 2))
    corner = cad.line(p1=(X / 2, X / 2), p0=(X / 2, X / 2))
    rightedge = cad.line(p1=(X / 2, 0), p0=(X / 2, X / 2))

    # Create IGA mesh object
    mesh = IGAMesh()

    # Create NURBS surfaces and add them
    sections = [0, 1 / 3, 2 / 3, 1]
    edges = [topedge, corner, rightedge]

    for i in range(len(sections) - 1):
        # Line sections
        osec = origin.slice(0, sections[i], sections[i + 1])
        bsec = burn.slice(0, sections[i], sections[i + 1])
        fsec = fuel.slice(0, sections[i], sections[i + 1])
        csec = clad.slice(0, sections[i], sections[i + 1])

        # Create patches
        mesh.add_patch(Patch(cad.ruled(osec, bsec), materials[0]))
        mesh.add_patch(Patch(cad.ruled(bsec, fsec), materials[1]))
        mesh.add_patch(Patch(cad.ruled(fsec, csec), materials[2]))
        mesh.add_patch(Patch(cad.ruled(csec, edges[i]), materials[3]))

    # Refine mesh
    mesh.refine(factor, degree)

    # Finalize mesh
    mesh.connect()

    # Set reflective boundary conditions
    mesh.set_reflective_conditions(("left", "bottom", "top", "right"))

    # Finalize mesh
    mesh.finalize()
    return mesh


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
    num_ordinates = 1024
    factor = 10
    degree = 2
    X = 1.36 / 2

    # Get XS data
    xs_server = kaist()

    # =====================================================================
    # Plot the problem
    # =====================================================================
    mesh = get_mesh(
        factor=3, degree=2, materials=["Displacer", "Fuel", "Cladding", "Water"]
    )
    print(mesh)

    # Plot mesh
    ax = mesh.plot(
        num_nodes=128,
        plot_ctrlpts=False,
        color_by="material",
        colors={
            "Displacer": "#800080",
            "Fuel": "#E69F00",
            "Cladding": "#ABABAB",
            "Water": "#0072B2",
        },
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
    plt.savefig("./figs/four_lobe.png", dpi=300)

    # =====================================================================
    # Get mesh we're actually going to use
    # =====================================================================
    mesh = get_mesh(
        factor=factor,
        degree=degree,
        materials=["Gas", "UO2 3%", "Guide Tube", "Water"],
    )
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
        return (
            mats.H.clone() + mats.B_out.clone() - mats.B_in.clone() - mats.S.clone()
        ).combine(), mats.F

    def mixed():
        return (
            (tts.H.clone() - tts.S.clone()).round(1e-5)
            + (mats.B_out.clone() - mats.B_in.clone()).combine()
        ), tts.F

    # =====================================================================
    # Solve each problem
    # =====================================================================
    solutions = {}
    for name, get_ops in zip(
        ["CSR", "Mixed"],
        [csr_only, mixed],
    ):
        T, F = get_ops()
        print(f"Total Compression: {T.compression}")
        psi, k = power(
            T=T,
            F=F,
            tol=1e-8,
            maxiter=1000,
            gpu_idx=0,
            lsoptions=LinearSolverOptions(restart=75, maxiter=10, tol=1e-8),
        )

        # Ravel solution back
        psi = psi.reshape(assembler.discretization)

        # Append solution
        solutions[name] = (psi, k)

    # Save solutions
    with open("solutions.pkl", "wb") as f:
        pickle.dump(solutions, f)

    # =====================================================================
    # Load OpenMC solution
    # =====================================================================
    # Get OpenMC solution
    k_mc = [1.256694399791112, 6.641896241106302e-05]
    phi_mc = np.load("./openmc/mesh_flux.npy")
    phi_mc_stdev = np.load("./openmc/mesh_stdev.npy")

    # Ensure OpenMC solution is normalized
    phi_mc_stdev /= np.linalg.norm(phi_mc.flatten(), 2)
    phi_mc /= np.linalg.norm(phi_mc.flatten(), 2)

    # Get mapping for mesh element averaging
    pids, coords = mesh.map_regular_mesh(shape=phi_mc.shape[1:], N=(5, 5))

    # =====================================================================
    # Calculate statistics
    # =====================================================================
    stats = {
        "methods": ["CSR", "Mixed", "TT", "TT (rounded)"],
        "k": {
            "error": [],
            "relative_error": [],
            "std_score": [],
        },
        "psi": {
            "minimum": [],
            "q1": [],
            "median": [],
            "q2": [],
            "maximum": [],
            "mean": [],
            "error": {
                "minimum": [],
                "q1": [],
                "median": [],
                "q2": [],
                "maximum": [],
                "mean": [],
            },
            "relative_error": {
                "minimum": [],
                "q1": [],
                "median": [],
                "q2": [],
                "maximum": [],
                "mean": [],
            },
            "zscore": {
                "minimum": [],
                "q1": [],
                "median": [],
                "q2": [],
                "maximum": [],
                "mean": [],
            },
            "l2 error": [],
        },
    }
    X, Y = np.meshgrid(
        np.linspace(0, X, phi_mc.shape[1]),
        np.linspace(0, X, phi_mc.shape[2]),
    )
    for name, solution in solutions.items():
        # Get k and psi
        psi, k = solution

        # Integrate angular component to get scalar flux
        phi = assembler.angular_integral(psi).numpy()
        phi /= np.linalg.norm(phi.flatten())

        # Eigenvalue statistics
        stats["k"]["error"].append((k - k_mc[0]) * 1e5)
        stats["k"]["relative_error"].append(
            abs(stats["k"]["error"][-1]) / (k_mc[0] * 1e5)
        )
        stats["k"]["std_score"].append(stats["k"]["error"][-1] / (k_mc[1] * 1e5))

        # Average on each mesh element in the regular mesh of OpenMC
        phi_avg = np.zeros(phi_mc.shape)
        for g in range(xs_server.num_groups):
            # Set control points
            mesh.set_phi(phi[g,])

            # Calculate regular mesh
            phi_avg[g,] = mesh.regular_mesh(pids, coords)

        # Normalize phi_avg
        phi_avg /= np.linalg.norm(phi_avg.flatten())

        for error_name, error in zip(
            ["error", "relative_error", "zscore"],
            [
                phi_avg - phi_mc,
                (phi_avg - phi_mc) / phi_mc,
                (phi_avg - phi_mc) / phi_mc_stdev,
            ],
        ):
            # Flatten error
            error = error.reshape((xs_server.num_groups, -1))

            # Calculate flux statistics
            minimum = np.zeros((xs_server.num_groups + 1))
            q1 = np.zeros((xs_server.num_groups + 1))
            median = np.zeros((xs_server.num_groups + 1))
            q2 = np.zeros((xs_server.num_groups + 1))
            maximum = np.zeros((xs_server.num_groups + 1))
            mean = np.zeros((xs_server.num_groups + 1))

            for g in range(xs_server.num_groups):
                minimum[g] = np.min(error[g,])
                q1[g] = np.percentile(error[g,], 25)
                median[g] = np.median(error[g,])
                q2[g] = np.percentile(error[g,], 75)
                maximum[g] = np.max(error[g,])
                mean[g] = np.mean(error[g,])

            minimum[-1] = np.min(minimum[:-1])
            q1[-1] = np.percentile(error, 25)
            median[-1] = np.median(error)
            q2[-1] = np.percentile(error, 75)
            maximum[-1] = np.max(maximum[:-1])
            mean[-1] = np.mean(error)

            # Add this to results
            stats["psi"][error_name]["minimum"].append(minimum)
            stats["psi"][error_name]["q1"].append(q1)
            stats["psi"][error_name]["median"].append(median)
            stats["psi"][error_name]["q2"].append(q2)
            stats["psi"][error_name]["maximum"].append(maximum)
            stats["psi"][error_name]["mean"].append(mean)

        # Calculate L2 error
        l2 = np.zeros((xs_server.num_groups + 1))
        error = (phi_avg - phi_mc).reshape((xs_server.num_groups, -1))
        for g in range(xs_server.num_groups):
            l2[g] = np.linalg.norm(error[g,]) / np.linalg.norm(phi_mc[g,].flatten())
        l2[-1] = np.linalg.norm(error.flatten()) / np.linalg.norm(phi_mc.flatten())

        # Add this to the results
        stats["psi"]["l2 error"].append(l2)

    # Save results
    with open("stats.pkl", "wb") as f:
        pickle.dump(stats, f)

    # =====================================================================
    # Plot results
    # =====================================================================
    psi, k = solutions[stats["methods"][np.argmin(stats["k"]["error"])]]

    # Integrate angular component to get scalar flux
    phi = assembler.angular_integral(psi).numpy()
    phi /= np.linalg.norm(phi.flatten())

    # Average on each mesh element in the regular mesh of OpenMC
    phi_avg = np.zeros(phi_mc.shape)
    for g in range(xs_server.num_groups):
        # Set control points
        mesh.set_phi(phi[g,])

        # Calculate regular mesh
        phi_avg[g,] = mesh.regular_mesh(pids, coords)

        # Plot
        plt.clf()
        ax, cbar = mesh.plot(plot_ctrlpts=False)
        cbar.set_label(f"$\\phi_{g + 1}" + "(\\hat{x}, \\hat{y})$")
        plt.tight_layout()
        plt.savefig(f"./figs/phi_{g + 1}.png", dpi=300)

    phi_avg /= np.linalg.norm(phi_avg.flatten())

    for error_name, error in zip(
        [
            "Scalar Flux Error",
            "Scalar Flux Relative Error",
            "Scalar Flux Number of Standard Deviations",
        ],
        [
            phi_avg - phi_mc,
            (phi_avg - phi_mc) / phi_mc,
            (phi_avg - phi_mc) / phi_mc_stdev,
        ],
    ):
        for g in range(xs_server.num_groups):
            plt.clf()
            ax = plt.gca()
            cmesh = ax.pcolormesh(X, Y, error[g,], cmap="plasma")
            divider = make_axes_locatable(ax)
            cbar = plt.colorbar(
                cmesh,
                cax=divider.append_axes("right", size="5%", pad=0.05),
            )
            cbar.set_label(error_name)
            ax.set_aspect("equal")
            ax.set_xlabel(r"$x(\hat{x}, \hat{y})~(cm)$")
            ax.set_ylabel(r"$y(\hat{x}, \hat{y})~(cm)$")
            plt.tight_layout()
            plt.savefig(f"./figs/{error_name}_{g + 1}.png", transparent=True, dpi=300)
