import multiprocessing
import pickle
import sys
import time
from typing import Literal, Tuple, Union

if __name__ == "__main__":
    multiprocessing.set_start_method("spawn")
    sys.path.append("../..")

import numpy as np
import torch as tn
from igakit import cad
from ttnte.assemblers import MatrixAssembler, TTAssembler
from ttnte.cad import Patch
from ttnte.iga import IGAMesh
from ttnte.linalg import LinearSolverOptions, TTOperator, cpp_available, power
from ttnte.xs.benchmarks import c5g7

from runner import Runner


def get_xs(num_groups: Literal[7]):
    """"""
    server = c5g7()
    assert server.num_groups == num_groups
    return server


def get_mesh(factor: Union[int, Tuple[int]], degree: Union[int, Tuple[int]]):
    """"""
    # Create quarter circle NURBS surface
    radius = 0.54  # cm
    pitch = 1.26  # cm
    c0 = cad.circle(radius=radius, angle=np.pi / 2)
    c1 = c0.slice(0, 0, 0.5)
    c2 = c0.slice(0, 0.5, 1)
    l0 = cad.line(p0=(0, 0), p1=(0, 0))

    # Create water patch
    l1 = cad.line(p0=(pitch / 2, 0), p1=(pitch / 2, pitch / 2))
    l2 = cad.line(p0=(pitch / 2, pitch / 2), p1=(0, pitch / 2))

    # Create NURBS surfaces
    fuel = [Patch(cad.ruled(l0, c1), "UO2"), Patch(cad.ruled(l0, c2), "UO2")]
    water = [Patch(cad.ruled(c1, l1), "Water"), Patch(cad.ruled(c2, l2), "Water")]

    # Initialize IGA mesh and add the patches
    mesh = IGAMesh(max_processes=32)
    for patch in fuel + water:
        mesh.add_patch(patch)

    # Refine each patch to have 6 knot spans with degree 2
    mesh.refine(factor, degree)

    # Connect patches
    mesh.connect()

    # Set reflective boundary conditions
    mesh.set_reflective_conditions(("left", "bottom", "top", "right"))

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
    num_ordinates = 1024
    factor = 10
    degree = 2
    eps = 1e-5

    # Power iteration options
    tol = 1e-8
    maxiter = 1000
    gpu_idx = 0
    verbose = True

    # GMRES options
    lsoptions = LinearSolverOptions(
        tol=1e-10,
        maxiter=10,
        restart=75,
        solve_method="batched",
        verbose=True,
    )

    # Get XS data
    xs_server = get_xs(7)

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
        "power": {"time": []},
        "psi": {"value": []},
        "k": {"value": []},
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
    tts = assembler.build(use_tt=False, eps=eps)

    # Save TT information
    assembler.save_info("./tt_info.csv")

    # Save data
    data["nelements"]["matrix"] = {
        "H": mats.H.nelements,
        "S": mats.S.nelements,
        "F": mats.F.nelements,
        "B_in": mats.B_in.nelements,
        "B_out": mats.B_out.nelements,
    }
    data["compression"]["matrix"] = {
        "H": mats.H.compression,
        "S": mats.S.compression,
        "F": mats.F.compression,
        "B_in": mats.B_in.compression,
        "B_out": mats.B_out.compression,
    }
    data["compression"]["tt"] = {
        "H": tts.H.compression,
        "S": tts.S.compression,
        "F": tts.F.compression,
        "B_in": tts.B_in.compression,
        "B_out": tts.B_out.compression,
    }
    data["ranks"] = {
        "H": tts.H.ranks,
        "S": tts.S.ranks,
        "F": tts.F.ranks,
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
        print(name)
        # Get total operator
        T, F = get_ops(mats, tts, eps), (tts.F if name != "CSR" else mats.F)
        print(f"Total Compression: {T.compression}")
        for op in T.operators:
            if isinstance(op, TTOperator):
                print(f"Ranks: {op.ranks}")

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

        # Run power iteration
        start = time.time()
        psi, k = power(
            T=T,
            F=F,
            tol=tol,
            maxiter=maxiter,
            gpu_idx=gpu_idx,
            lsoptions=lsoptions,
            verbose=verbose,
        )
        data["power"]["time"].append(time.time() - start)

        # Ravel solution back
        psi = psi.reshape(assembler.discretization)

        # Append data
        data["k"]["value"].append(k)
        data["psi"]["value"].append(psi.numpy())

        # Save data
        with open("data.pkl", "wb") as f:
            pickle.dump(data, f)
