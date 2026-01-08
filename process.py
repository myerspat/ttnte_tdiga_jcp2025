import json
import pickle
from pathlib import Path
from typing import Callable, List, Optional, Union

import numpy as np
import torch as tn
from torchtt import TT
from ttnte.assemblers import MatrixAssembler


def process(
    get_xs: Callable,
    get_mesh: Callable,
    dir: Union[str, Path],
    mc_leakage_frac: List[float],
    mc_solution: Optional[List[np.ndarray]] = None,
):
    # Make sure we have a path
    dir = Path(dir)

    # Get JSON (Lines) file
    print(dir)
    jsonl_file = list(dir.rglob("*.jsonl"))[0]

    # Get regular mesh object
    regular_mesh = None

    i = 0
    phi_csr = {}
    phi_csr_avg = {}
    with open(jsonl_file, "r", encoding="utf-8") as infile, open(
        jsonl_file.with_name(f"processed_{jsonl_file.name}"), "w"
    ) as outfile:
        # Iterate through lines
        for line in infile:
            # Parse JSON data
            result = json.loads(line)

            case = [
                result["num_ordinates"],
                result["num_groups"],
                (
                    result["factor"]
                    if isinstance(result["factor"], int)
                    else result["factor"][0]
                )
                + (
                    result["degree"]
                    if isinstance(result["degree"], int)
                    else result["degree"][0]
                ),
                (
                    result["factor"]
                    if isinstance(result["factor"], int)
                    else result["factor"][1]
                )
                + (
                    result["degree"]
                    if isinstance(result["degree"], int)
                    else result["degree"][1]
                ),
                (
                    result["degree"]
                    if isinstance(result["degree"], int)
                    else result["degree"][0]
                ),
                (
                    result["degree"]
                    if isinstance(result["degree"], int)
                    else result["degree"][1]
                ),
                result["eps"],
                result["device"],
            ]

            print(
                "Running case {}: N={}, G={}, A={}, B={}, p={}, q={}, eps={}, device={}".format(
                    i, *case
                )
            )
            i += 1

            # Get XSs and mesh
            xs_server = get_xs(result["num_groups"])
            mesh = get_mesh(result["factor"], result["degree"])

            # Map regular mesh if needed
            if regular_mesh == None and mc_solution is not None:
                regular_mesh = mesh.map_regular_mesh(
                    shape=mc_solution[0].shape[1:], N=(5, 5)
                )

            # Make a matrix assembler
            assembler = MatrixAssembler(
                mesh=mesh, xs_server=xs_server, num_ordinates=result["num_ordinates"]
            )

            # Read in solution data
            solution = pickle.load(
                open(
                    dir / "meshes/N{}_G{}_A{}_B{}_p{}_q{}_eps{}{}.pkl".format(*case),
                    "rb",
                )
            )
            # Calculate leakage fractions
            result["leakage_fraction"] = {
                "value": [],
                "error": [],
                "zscore": [],
            }
            result["flux_stats"] = {
                "ranks": [],
                "compression": [],
                "l2 error (pointwise)": [],
            }

            if mc_solution is not None:
                result["flux_stats"] = {
                    "minimum": [],
                    "q1": [],
                    "median": [],
                    "q2": [],
                    "maximum": [],
                    "mean": [],
                    "l2 error": [],
                    "l2 error (pointwise)": [],
                    "l2 error to csr": [],
                    "ranks": [],
                    "compression": [],
                }

            for name, psi in solution.items():
                # Get leakage fraction data
                result["leakage_fraction"]["value"].append(
                    float(
                        assembler.outward_current(
                            tn.tensor(psi.reshape(assembler.discretization))
                        )
                        / assembler.total_production()
                    )
                )
                result["leakage_fraction"]["error"].append(
                    result["leakage_fraction"]["value"][-1] - mc_leakage_frac[0]
                )
                result["leakage_fraction"]["zscore"].append(
                    result["leakage_fraction"]["error"][-1] / mc_leakage_frac[1]
                )

                tt = TT(
                    np.squeeze(psi.reshape(assembler.discretization)),
                    eps=result["eps"] if name != "CSR" else 0,
                )
                result["flux_stats"]["ranks"].append(np.array(tt.R[1:-1]).tolist())
                result["flux_stats"]["compression"].append(
                    float(np.prod(psi.shape) / sum(tn.numel(c) for c in tt.cores))
                )

                # Get scalar flux
                phi = assembler.angular_integral(
                    tn.tensor(psi.reshape(assembler.discretization))
                ).reshape(assembler.discretization[-4:])

                phi_pointwise = np.empty(
                    (xs_server.num_groups, mesh.num_patches, 128, 128)
                )
                l2error = []
                key = tuple(
                    [
                        result[key]
                        for key in [
                            "num_ordinates",
                            "num_groups",
                            "factor",
                            "degree",
                        ]
                    ]
                )
                for g in range(xs_server.num_groups):
                    mesh.set_phi(phi[0,])

                    X, Y = np.meshgrid(np.linspace(0, 1, 128), np.linspace(0, 1, 128))

                    phi_pointwise[g,] = np.array(
                        [
                            c[:, -1]
                            for c in mesh(
                                np.concatenate(
                                    [X.reshape((-1, 1)), Y.reshape((-1, 1))], axis=1
                                )
                            )
                        ]
                    ).reshape((mesh.num_patches, 128, 128))

                    if name == "CSR":
                        continue
                    else:
                        assert isinstance(phi_csr[key], np.ndarray)
                        l2error.append(
                            np.linalg.norm(
                                (phi_pointwise[g,] - phi_csr[key][g,]).flatten(), 2
                            )
                            / np.linalg.norm(phi_csr[key][g,].flatten(), 2)
                        )

                if name == "CSR":
                    phi_csr[key] = phi_pointwise

                else:
                    assert key in phi_csr
                    assert isinstance(phi_csr[key], np.ndarray)
                    l2error.append(
                        np.linalg.norm((phi_pointwise - phi_csr[key]).flatten(), 2)
                        / np.linalg.norm(phi_csr[key].flatten(), 2)
                    )
                    result["flux_stats"]["l2 error (pointwise)"].append(l2error)

                # Get stats for solution vector
                if mc_solution is not None:
                    assert regular_mesh is not None

                    # Get scalar flux
                    phi = assembler.angular_integral(
                        tn.tensor(psi.reshape(assembler.discretization))
                    ).reshape(assembler.discretization[-4:])

                    # Get phi average
                    error = np.empty(mc_solution[0].shape)

                    if name == "CSR":
                        phi_csr_avg[key] = np.zeros((xs_server.num_groups, 128, 128))

                    # Calculate stats comparing to MC solution
                    minimum = []
                    q1 = []
                    median = []
                    q2 = []
                    maximum = []
                    mean = []
                    l2 = []
                    l2tocsr = []

                    for g in range(xs_server.num_groups):
                        # Set control points
                        mesh.set_phi(phi[g,])

                        sol = mesh.regular_mesh(*regular_mesh)

                        # Set phi for mesh
                        error[g,] = sol - mc_solution[0][g,]

                        z = np.abs(error[g,] / mc_solution[1][g,])
                        minimum.append(np.min(z))
                        q1.append(np.percentile(z, 25))
                        median.append(np.median(z))
                        q2.append(np.percentile(z, 75))
                        maximum.append(np.max(z))
                        mean.append(np.mean(z))
                        l2.append(
                            np.linalg.norm(error[g,].flatten(), 2)
                            / np.linalg.norm(mc_solution[0][g,].flatten(), 2)
                        )

                        if name == "CSR":
                            assert key in phi_csr_avg
                            assert isinstance(phi_csr_avg[key], np.ndarray)
                            phi_csr_avg[key][g,] = sol

                        elif not (phi_csr_avg[key] == 0).all():
                            assert key in phi_csr_avg
                            assert isinstance(phi_csr_avg[key], np.ndarray)
                            l2tocsr.append(
                                np.linalg.norm(
                                    (sol - phi_csr_avg[key][g,]).flatten(), 2
                                )
                                / np.linalg.norm(phi_csr_avg[key][g,].flatten(), 2)
                            )

                    z = np.abs(error / mc_solution[1])
                    minimum.append(np.min(z))
                    q1.append(np.percentile(z, 25))
                    median.append(np.median(z))
                    q2.append(np.percentile(z, 75))
                    maximum.append(np.max(z))
                    mean.append(np.mean(z))
                    l2.append(
                        np.linalg.norm(error.flatten(), 2)
                        / np.linalg.norm(mc_solution[0].flatten(), 2)
                    )

                    if name != "CSR":
                        assert key in phi_csr_avg
                        assert (
                            isinstance(phi_csr_avg[key], np.ndarray)
                            and not (phi_csr_avg[key] == 0).all()
                        )
                        l2tocsr.append(np.sqrt(np.sum(np.array(l2tocsr) ** 2)))

                    # Add stats to total result
                    result["flux_stats"]["minimum"].append(minimum)
                    result["flux_stats"]["q1"].append(q1)
                    result["flux_stats"]["median"].append(median)
                    result["flux_stats"]["q2"].append(q2)
                    result["flux_stats"]["maximum"].append(maximum)
                    result["flux_stats"]["mean"].append(mean)
                    result["flux_stats"]["l2 error"].append(l2)
                    result["flux_stats"]["l2 error to csr"].append(l2tocsr)

            print(result)

            outfile.write(json.dumps(result) + "\n")
