import os
import gc
import json
import time
import pickle
import itertools
from typing import Union, List, Tuple, Optional, Callable
from pathlib import Path

import torch as tn
import numpy as np

from ttnte.linalg import (
    LinearSolverOptions,
    gmres,
    LinearOperator,
    SparseOperator,
    ScatterOperator,
    TTOperator,
)
from ttnte.xs import Server
from ttnte.assemblers import MatrixAssembler, TTAssembler, OperatorData
from ttnte.iga import IGAMesh


class Runner:
    configs = ["CSR", "TT", "Mixed", "TT (rounded)", "Mixed (rounded)"]

    def __init__(
        self,
        study_name: str,
        study_path: Union[str, Path],
        num_ordinates: List[int],
        num_groups: List[int],
        factors: List[Union[int, Tuple[int]]],
        degrees: List[Union[int, Tuple[int]]],
        eps: List[float],
        gpu_idx: Optional[int] = None,
        cpu_and_gpu: bool = True,
        verbose: bool = True,
    ):
        """"""
        self._study_name = study_name
        self._study_path = Path(os.path.abspath(study_path))
        self._num_ordinates = num_ordinates
        self._num_groups = num_groups
        self._factors = factors
        self._degrees = degrees
        self._eps = np.sort(eps, axis=None).tolist()
        self._devices = ([None] if cpu_and_gpu or gpu_idx is None else []) + (
            [gpu_idx] if gpu_idx is not None else []
        )
        self._verbose = verbose

        # Path to meshes
        self._mesh_dir = self._study_path / "meshes"
        self._rnorm_dir = self._study_path / "rnorms"

        self._methods = {
            "CSR": self._pureCSR,
            "TT": self._pureTT,
            "Mixed": self._mixed,
            "TT (rounded)": self._pureTTrounded,
            "Mixed (rounded)": self._mixed_rounded,
        }

    # ======================================================================
    # Methods

    def run(self, get_xs: Callable, get_mesh: Callable, lsoptions: LinearSolverOptions):
        """"""
        if self._verbose:
            print(
                "Running {} combinations".format(
                    len(self._num_ordinates)
                    * len(self._num_groups)
                    * len(self._factors)
                    * len(self._degrees)
                    * len(self._eps)
                    * len(self._devices)
                    * 3
                    + len(self._num_ordinates)
                    * len(self._num_groups)
                    * len(self._factors)
                    * len(self._degrees)
                    * len(self._devices)
                )
            )

        # Create directories
        self._study_path.mkdir(parents=True, exist_ok=True)
        self._mesh_dir.mkdir(parents=True, exist_ok=True)
        self._rnorm_dir.mkdir(parents=True, exist_ok=True)

        # Reset carry data
        self._mats = None
        self._tts = None

        # Time the total run
        self._start = time.time()
        self._idx = 0
        self._last_config = None
        xs_server = None
        mesh = None

        for num_groups, num_ordinates, factor, degree in itertools.product(
            self._num_groups, self._num_ordinates, self._factors, self._degrees
        ):
            if (
                self._last_config == None
                or self._last_config["num_ordinates"] != num_ordinates
                or self._last_config["factor"] != factor
                or self._last_config["degree"] != degree
                or self._last_config["num_groups"] != num_groups
            ):
                # Generate new mesh and XSs
                xs_server = get_xs(num_groups)
                mesh = get_mesh(factor, degree)

                self._mats = None
                self._tts = None
                gc.collect()

            for eps in self._eps:
                for device in self._devices:
                    assert isinstance(xs_server, Server)
                    assert isinstance(mesh, IGAMesh)

                    start = time.time()
                    try:
                        lsoptions.gpu_idx = device
                        lsoptions.solve_method = (
                            "incremental" if device is None else "batched"
                        )
                        results = self._run_case(
                            num_ordinates,
                            factor,
                            degree,
                            eps,
                            xs_server,
                            mesh,
                            lsoptions,
                        )

                        if results is not None:
                            # Get path for solution and rnorms
                            spath = self._solution_path(
                                num_ordinates,
                                num_groups,
                                factor,
                                degree,
                                eps,
                                device,
                            )
                            rpath = self._rnorm_path(
                                num_ordinates,
                                num_groups,
                                factor,
                                degree,
                                eps,
                                device,
                            )
                            self._write_json(results[0])
                            self._write_pickle(spath, results[1])
                            self._write_pickle(rpath, results[2])

                            print(
                                "Finished iteration! Run time = {:.3f} s, Elapsed time = {:.3f}".format(
                                    time.time() - start, time.time() - self._start
                                )
                            )

                    except Exception as e:
                        print(
                            "Failed to run case {} with run time {} s with error:\n{}".format(
                                self._idx, time.time() - start, e
                            )
                        )

                    self._last_config = {
                        "num_groups": num_groups,
                        "num_ordinates": num_ordinates,
                        "factor": factor,
                        "degree": degree,
                        "eps": eps,
                        "device": device,
                    }
                    # Reset TTs
                    self._tts = None
                    self._idx += 1
                    gc.collect()

    def _run_case(
        self,
        num_ordinates,
        factor,
        degree,
        eps,
        xs_server: Server,
        mesh: IGAMesh,
        lsoptions: LinearSolverOptions,
    ):
        """"""
        if self._verbose:
            print(80 * "=")

        # Get path to solution file
        solution_path = self._solution_path(
            num_ordinates, xs_server.num_groups, factor, degree, eps, lsoptions.gpu_idx
        )

        # Check if file exists
        if solution_path.is_file():
            print(
                "Already ran case {}: N={}, G={}, A={}, B={}, p={}, q={}, eps={}, device={}".format(
                    self._idx,
                    *self.case_name(
                        num_ordinates,
                        xs_server.num_groups,
                        factor,
                        degree,
                        eps,
                        lsoptions.gpu_idx,
                    ),
                )
            )
            return None
        else:
            print(
                "Running case {}: N={}, G={}, A={}, B={}, p={}, q={}, eps={}, device={}".format(
                    self._idx,
                    *self.case_name(
                        num_ordinates,
                        xs_server.num_groups,
                        factor,
                        degree,
                        eps,
                        lsoptions.gpu_idx,
                    ),
                )
            )

        # Start data collection
        result = {
            "num_ordinates": num_ordinates,
            "num_groups": xs_server.num_groups,
            "factor": factor,
            "degree": degree,
            "eps": eps,
            "device": ("cpu" if lsoptions.gpu_idx is None else "gpu"),
            "nelements": {"total": []},
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
        }
        solutions = {}
        rnorms = {}

        # Build matrices and save build info if needed
        if self._mats is None:
            start = time.time()
            try:
                assembler = MatrixAssembler(
                    mesh=mesh,
                    xs_server=xs_server,
                    num_ordinates=num_ordinates,
                    max_processes=1,
                )
                self._mats = assembler.build()
                result["matrix_assembly_time"] = time.time() - start

                assert isinstance(self._mats.H, SparseOperator)
                assert isinstance(self._mats.S, ScatterOperator)
                assert isinstance(self._mats.B_out, SparseOperator)

                # Save info for assembler
                result["avg_element_size"] = assembler.avg_element_size.item()
                result["nnz"] = {"H": self._mats.H.nnz, "B_out": self._mats.B_out.nnz}
                result["nelements"]["matrix"] = {
                    "H": self._mats.H.nelements,
                    "B_out": self._mats.B_out.nelements,
                    "S": self._mats.S.nelements,
                }
                result["compression"]["matrix"] = {
                    "H": self._mats.H.compression,
                    "B_out": self._mats.B_out.compression,
                    "S": self._mats.S.compression,
                }

                if isinstance(self._mats.B_in, SparseOperator):
                    result["nnz"]["B_in"] = self._mats.B_in.nnz
                    result["nelements"]["matrix"]["B_in"] = self._mats.B_in.nelements
                    result["compression"]["matrix"][
                        "B_in"
                    ] = self._mats.B_in.compression

                del assembler
            except Exception as e:
                print(f"Failed to build matrices with error {e}")
                del self._mats
                gc.collect()
                self._mats = None

        # Build TTs and save build info if needed
        try:
            if self._tts is None:
                start = time.time()
                assembler = TTAssembler(
                    mesh=mesh,
                    xs_server=xs_server,
                    num_ordinates=num_ordinates,
                    max_processes=1,
                )
                self._tts = assembler.build(use_tt=False, eps=eps, q=False)
                result["tt_assembly_time"] = time.time() - start
                del assembler

            elif self._last_config is not None and self._last_config["eps"] != eps:
                assert isinstance(self._tts.H, TTOperator)
                assert isinstance(self._tts.S, TTOperator)
                assert isinstance(self._tts.B_out, TTOperator)

                # TTs are already made we just need to round them
                self._tts.H = self._tts.H.round(eps)
                self._tts.S = self._tts.S.round(eps)
                self._tts.B_out = self._tts.B_out.round(eps)

                if isinstance(self._tts.B_in, TTOperator):
                    self._tts.B_in = self._tts.B_in.round(eps)

            assert isinstance(self._mats, OperatorData) and isinstance(
                self._tts, OperatorData
            )
            assert isinstance(self._mats.q, tn.Tensor)
            assert isinstance(self._tts.H, TTOperator)
            assert isinstance(self._tts.S, TTOperator)
            assert isinstance(self._tts.B_out, TTOperator)
            gc.collect()

            result["nelements"]["tts"] = {
                "H": self._tts.H.nelements,
                "B_out": self._tts.B_out.nelements,
                "S": self._tts.S.nelements,
            }
            result["compression"]["tts"] = {
                "H": self._tts.H.compression,
                "B_out": self._tts.B_out.compression,
                "S": self._tts.S.compression,
            }
            result["ranks"] = {
                "H": self._tts.H.ranks,
                "S": self._tts.S.ranks,
                "B_out": self._tts.B_out.ranks,
            }
            if isinstance(self._tts.B_in, TTOperator):
                result["nelements"]["tts"]["B_in"] = self._tts.B_in.nelements
                result["compression"]["tts"]["B_in"] = self._tts.B_in.compression
                result["ranks"]["B_in"] = self._tts.B_in.ranks

        except Exception as e:
            print(f"Failed to build TT operators with error {e}")
            del self._tts
            gc.collect()
            self._tts = None

        if self._mats == None and self._tts == None:
            raise RuntimeError("Operators failed to build")

        # Iterate through each case
        for name in self.configs:
            # Skip whats already been done
            if name == "CSR" and eps != self._eps[0]:
                continue

            subresult = {
                "solve_method": name,
                "nelements": None,
                "compression": None,
                "nnz": None,
                "ranks": None,
                "matvec": {
                    "time": None,
                    "stdev": None,
                },
                "gmres": {
                    "time": None,
                    "rnorm": None,
                    "converged": None,
                },
            }

            try:
                # Get total operator
                T = self._methods[name](self._mats, self._tts, eps)

                # Add total operator data
                subresult["nelements"] = T.nelements
                subresult["compression"] = T.compression

                if name == "CSR":
                    assert len(T.operators) == 2
                    subresult["nnz"] = T.operators[-1].nnz

                if name == "TT (rounded)":
                    assert len(T.operators) == 1
                    subresult["ranks"] = T.operators[0].ranks

                # Run total operator through matvecs to get timing results
                if lsoptions.gpu_idx is not None:
                    T.cuda(lsoptions.gpu_idx)
                times = np.zeros(1000, dtype=np.float64)
                vec = tn.rand(*T.input_shape, dtype=tn.float64, device=T.device)

                for i in range(times.size):
                    start = time.time()
                    _ = T @ vec
                    times[i] = time.time() - start

                subresult["matvec"]["time"] = np.average(times)
                subresult["matvec"]["stdev"] = np.std(times)
                if lsoptions.gpu_idx is not None:
                    T.cpu()
                del vec
                gc.collect()

                # Run and time GMRES
                start = time.time()
                psi, rnorm = self._run_gmres(A=T, b=self._mats.q, lsoptions=lsoptions)
                subresult["gmres"]["time"] = time.time() - start
                gc.collect()

                # Add psi and norms
                solutions[name] = psi.numpy()
                rnorms[name] = rnorm.numpy()

                # Get GMRES info
                assert isinstance(self._mats.q, tn.Tensor)
                subresult["gmres"]["rnorm"] = rnorm[-1].item()
                subresult["gmres"]["converged"] = (
                    True
                    if rnorm[-1]
                    < (max(lsoptions.tol * self._mats.q.norm(2).item(), lsoptions.atol))
                    else False
                )
                gc.collect()

                # Add results to total result
                result["solve_method"].append(subresult["solve_method"])
                result["nelements"]["total"].append(subresult["nelements"])
                result["compression"]["total"].append(subresult["compression"])
                result["matvec"]["time"].append(subresult["matvec"]["time"])
                result["matvec"]["stdev"].append(subresult["matvec"]["stdev"])
                result["gmres"]["time"].append(subresult["gmres"]["time"])
                result["gmres"]["rnorm"].append(subresult["gmres"]["rnorm"])
                result["gmres"]["converged"].append(subresult["gmres"]["converged"])

                if subresult["nnz"] is not None:
                    if "nnz" in result:
                        result["nnz"]["T"] = subresult["nnz"]
                    else:
                        result["nnz"] = {"T": subresult["nnz"]}

                if subresult["ranks"] is not None:
                    if "ranks" in result:
                        result["ranks"]["T"] = subresult["ranks"]
                    else:
                        result["ranks"] = {"T": subresult["ranks"]}

            except Exception as e:
                print(f"Config {name} failed with exception {e}")

                # Clear problematic operators if needed
                if name in ("CSR", "Mixed", "Mixed (rounded)"):
                    del self._mats
                    gc.collect()
                    self._mats = None

                if name in ("TT", "Mixed", "TT (rounded)", "Mixed (rounded)"):
                    del self._tts
                    gc.collect()
                    self._tts = None

        return result, solutions, rnorms

    def _solution_path(self, num_ordinates, num_groups, factor, degree, eps, device):
        """"""
        name = "N{}_G{}_A{}_B{}_p{}_q{}_eps{}".format(
            *self.case_name(num_ordinates, num_groups, factor, degree, eps, device)
        ) + ("cpu" if device is None else "gpu")

        return self._mesh_dir / (name + ".pkl")

    def _rnorm_path(self, num_ordinates, num_groups, factor, degree, eps, device):
        """"""
        name = "N{}_G{}_A{}_B{}_p{}_q{}_eps{}".format(
            *self.case_name(num_ordinates, num_groups, factor, degree, eps, device)
        ) + ("cpu" if device is None else "gpu")

        return self._rnorm_dir / (name + ".pkl")

    def _write_json(self, result):
        """"""
        with open(
            self._study_path / (self._study_name + ".jsonl"), "a", encoding="utf-8"
        ) as f:
            f.write(json.dumps(result) + "\n")

    def _write_pickle(self, path, data):
        """"""
        with open(path, "wb") as f:
            pickle.dump(data, f)

    # ======================================================================
    # Static methods

    @staticmethod
    def case_name(num_ordinates, num_groups, factor, degree, eps, device):
        """"""
        return [
            num_ordinates,
            num_groups,
            (factor if isinstance(factor, int) else factor[0])
            + (degree if isinstance(degree, int) else degree[0]),
            (factor if isinstance(factor, int) else factor[1])
            + (degree if isinstance(degree, int) else degree[1]),
            degree if isinstance(degree, int) else degree[0],
            degree if isinstance(degree, int) else degree[1],
            eps,
            "cpu" if device is None else "gpu",
        ]

    @staticmethod
    def _run_gmres(
        A: LinearOperator, b: tn.Tensor, lsoptions: LinearSolverOptions
    ) -> Tuple[tn.Tensor, tn.Tensor]:
        """"""
        return gmres(
            A=A,
            b=b,
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

    @staticmethod
    def _pureCSR(mats: OperatorData, tts: OperatorData, eps: float) -> LinearOperator:
        """"""
        assert mats.H is not None
        assert mats.B_out is not None
        assert mats.S is not None
        return (
            mats.H
            - mats.S
            + ((mats.B_out - mats.B_in) if mats.B_in is not None else (mats.B_out))
        ).combine()

    @staticmethod
    def _pureTT(mats: OperatorData, tts: OperatorData, eps: float) -> LinearOperator:
        """"""
        assert tts.H is not None
        assert tts.B_out is not None
        assert tts.S is not None
        return (
            tts.H.clone()
            - tts.S.clone()
            + (
                (tts.B_out.clone() - tts.B_in.clone())
                if tts.B_in is not None
                else (tts.B_out.clone())
            )
        )

    @staticmethod
    def _pureTTrounded(
        mats: OperatorData, tts: OperatorData, eps: float
    ) -> LinearOperator:
        """"""
        return Runner._pureTT(mats, tts, eps).round(eps)

    @staticmethod
    def _mixed(mats: OperatorData, tts: OperatorData, eps: float):
        """"""
        assert tts.H is not None
        assert mats.B_out is not None
        assert tts.S is not None
        return (
            tts.H
            - tts.S
            + (
                (mats.B_out - mats.B_in).combine()
                if mats.B_in is not None
                else (mats.B_out)
            )
        )

    @staticmethod
    def _mixed_rounded(mats: OperatorData, tts: OperatorData, eps: float):
        """"""
        assert tts.H is not None
        assert mats.B_out is not None
        assert tts.S is not None
        return (tts.H - tts.S).round(eps) + (
            (mats.B_out - mats.B_in).combine()
            if mats.B_in is not None
            else (mats.B_out)
        )
