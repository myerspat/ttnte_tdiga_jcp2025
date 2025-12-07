import multiprocessing
import os
import pickle
import sys
import time
from pathlib import Path
from typing import Tuple, Union

if __name__ == "__main__":
    multiprocessing.set_start_method("spawn")
    sys.path.append("../..")

import numpy as np
import torch as tn
from igakit import cad
from ttnte.assemblers import MatrixAssembler, TTAssembler
from ttnte.cad import Patch
from ttnte.cad.curves import qtrlobe
from ttnte.iga import IGAMesh
from ttnte.linalg import LinearSolverOptions, cpp_available, gmres
from ttnte.sources import IsotropicInternalSource
from ttnte.xs.benchmarks import Server

from runner import Runner


def get_xs(num_groups: int):
    """"""
    server = Server(
        {
            "Source": {
                "total": np.array([0.01]),
                "scatter_gtg": np.array([[[0.008]]]),
            },
            "Void": {
                "total": np.array([0]),
                "scatter_gtg": np.array([[[0]]]),
            },
            "Shield": {
                "total": np.array([3]),
                "scatter_gtg": np.array([[[0.5]]]),
            },
        }
    )
    assert server.num_groups == num_groups
    return server


def get_mesh(factor: Union[int, Tuple[int]], degree: Union[int, Tuple[int]]):
    # Initialize dimensional variables
    X = 10  # Channel pitch

    # Cruciform
    R = 2  # Radius defining valleys of fixed source
    delta = 1  # Width of lobes
    d2 = delta * 0.5  # Half width of lobes
    x = 0.25  # Portrusion of lobes

    # Shielding
    I = 3.75  # Inner radius
    O = 4.5  # Outer radius

    # NURBS curves
    origin = cad.line(p0=(0, 0), p1=(0, 0))
    cruciform = qtrlobe(outrad=R, portrs=x, hfwidth=d2)
    circleI = cad.circle(radius=I, angle=[np.pi / 2, 0])
    circleO = cad.circle(radius=O, angle=[np.pi / 2, 0])
    topedge = cad.line(p0=(0, X / 2), p1=(X / 2, X / 2))
    corner = cad.line(p1=(X / 2, X / 2), p0=(X / 2, X / 2))
    rightedge = cad.line(p1=(X / 2, 0), p0=(X / 2, X / 2))

    # Create IGA mesh object
    mesh = IGAMesh(max_processes=32)

    # Create and add NURBS surfaces
    sections = [0, 1 / 3, 2 / 3, 1]
    edges = [topedge, corner, rightedge]

    for i in range(len(sections) - 1):
        # Line sections
        csec = origin.slice(0, sections[i], sections[i + 1])
        ssec = cruciform.slice(0, sections[i], sections[i + 1])
        isec = circleI.slice(0, sections[i], sections[i + 1])
        osec = circleO.slice(0, sections[i], sections[i + 1])

        # Create source patch
        source = Patch(cad.ruled(csec, ssec), "Source")
        source.set_source(IsotropicInternalSource(np.ones((1, *source.shape))))
        mesh.add_patch(source)

        # Add remaining
        mesh.add_patch(Patch(cad.ruled(ssec, isec), "Void"))
        mesh.add_patch(Patch(cad.ruled(isec, osec), "Shield"))
        mesh.add_patch(Patch(cad.ruled(osec, edges[i]), "Void"))

    # Refine mesh resolution
    mesh.refine(factor=factor, degree=degree)

    # Connect patches
    mesh.connect()

    # Set reflective boundary conditions
    mesh.set_reflective_conditions(("left", "bottom"))

    # Finalize mesh
    mesh.finalize()
    print(mesh)
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
    num_ordinates = 4096
    factor = 10
    degree = 2
    eps = 1e-5

    # GMRES options
    lsoptions = LinearSolverOptions(
        gpu_idx=0,
        tol=1e-6,
        maxiter=1000,
        restart=100,
        solve_method="batched",
        verbose=True,
    )

    # Get XS data
    xs_server = get_xs(1)

    # Get mesh
    mesh = get_mesh(factor=factor, degree=degree)

    # Save data
    data = {
        "num_ordinates": num_ordinates,
        "num_groups": xs_server.num_groups,
        "num_patches": mesh.num_patches,
        "factor": factor,
        "degree": degree,
        "eps": eps,
        "device": "GPU",
        "nelements": {},
        "compression": {"total": []},
        "solve_method": [],
        "matvec": {
            "time": [],
            "stdev": [],
        },
        "gmres": {
            "rnorm": [],
            "time": [],
            "converged": [],
        },
        "psi": {"value": []},
        "rnorm": [],
    }

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
    tts = assembler.build(use_tt=False, eps=eps, q=False)

    # Save TT information
    assembler.save_info("./tt_info.csv")

    # Save data
    data["nelements"]["matrix"] = {
        "H": mats.H.nelements,
        "S": mats.S.nelements,
        "B_in": mats.B_in.nelements,
        "B_out": mats.B_out.nelements,
    }
    data["compression"]["matrix"] = {
        "H": mats.H.compression,
        "S": mats.S.compression,
        "B_in": mats.B_in.compression,
        "B_out": mats.B_out.compression,
    }
    data["compression"]["tt"] = {
        "H": tts.H.compression,
        "S": tts.S.compression,
        "B_in": tts.B_in.compression,
        "B_out": tts.B_out.compression,
    }
    data["ranks"] = {
        "H": tts.H.ranks,
        "S": tts.S.ranks,
        "B_in": tts.B_in.ranks,
        "B_out": tts.B_out.ranks,
    }

    # =====================================================================
    # Solve each problem
    # =====================================================================
    solutions = {}
    for name, get_ops in zip(
        ["CSR", "Mixed", "Mixed (rounded)"],
        [Runner._pureCSR, Runner._mixed, Runner._mixed_rounded],
    ):
        # Get total operator
        T = get_ops(mats, tts, eps)
        print(f"Total Compression: {T.compression}")

        # Add data
        data["solve_method"].append(name)
        data["compression"]["total"].append(T.compression)

        # Run total operator apply
        if lsoptions.gpu_idx != None:
            T.cuda(lsoptions.gpu_idx)
        times = np.zeros(1000, dtype=np.float64)
        vec = tn.rand(*T.input_shape, dtype=tn.float64, device=T.device).reshape(
            (-1, 1)
        )

        for i in range(times.size):
            start = time.time()
            _ = T @ vec
            times[i] = time.time() - start

        data["matvec"]["time"].append(np.average(times))
        data["matvec"]["stdev"].append(np.std(times))
        if lsoptions.gpu_idx != None:
            T.cpu()

        # Run solver
        start = time.time()
        psi, rnorm = gmres(
            A=T,
            b=mats.q,
            gpu_idx=lsoptions.gpu_idx,
            tol=lsoptions.tol,
            atol=lsoptions.atol,
            restart=lsoptions.restart,
            maxiter=lsoptions.maxiter,
            solve_method=lsoptions.solve_method,
            callback=lsoptions.callback,
            callback_frequency=lsoptions.callback_frequency,
            verbose=lsoptions.verbose,
        )
        data["gmres"]["time"].append(time.time() - start)
        data["gmres"]["rnorm"].append(rnorm[-1])
        data["gmres"]["converged"] = (
            True
            if rnorm[-1] < (max(lsoptions.tol * mats.q.norm(2).item(), lsoptions.atol))
            else False
        )

        # Ravel solution back
        psi = psi.reshape(assembler.discretization)

        # Append data
        data["psi"]["value"].append(psi.numpy())
        data["rnorm"].append(rnorm.numpy())

        # Save data
        with open("data.pkl", "wb") as f:
            pickle.dump(data, f)
