import multiprocessing
import pickle
import sys
import time
from typing import Literal

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
from ttnte.linalg import LinearSolverOptions, TTOperator, cpp_available, power
from ttnte.xs.benchmarks import kaist

from runner import Runner


def get_xs(num_groups: Literal[7]):
    """"""
    server = kaist()
    assert server.num_groups == num_groups
    return server


def get_mesh(
    factor, degree, materials=["BA (UO2 FA)", "UO2 3%", "Guide Tube", "Water"]
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
    mesh = IGAMesh(max_processes=32)

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
        max_processes=4,
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
