#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
PyTorch implementation of Robust Non-negative Tensor Factorization.

See docstring for robust_ntf() for a description of overall functionality.

Contents
--------
    robust_ntf() : computes Robust Non-negative Tensor Factorization of a given
                   input tensor.
    initialize_rntf() : provides initial estimates for factor matrices and the
                        outlier tensor.
    update_factor() : multiplicatively updates the current estimate of a given
                      factor matrix.
    update_outlier() : multiplicatively updates the current estimate of a given
                       matricized outlier tensor.

If you find bugs and/or limitations, please email neel DOT dey AT nyu DOT edu.

Created March 2019, refactored September 2019.

Enhancements added by William Warriner August 2020.
"""

from typing import Callable, List, Tuple, Union, Dict, Any, Optional
from pathlib import Path, PurePath
import pickle
import json

import pandas as pd
import numpy as np
import torch
from torch.nn.functional import normalize
import tensorly as tl

from .foldings import folder, unfolder
from .matrix_utils import kr_bcd, beta_divergence, L21_norm

tl.set_backend("pytorch")

PathLike = Union[str, Path, PurePath]

# TODO fail_on_cpu
# TODO pull out loop into separate function so we can store intermediate results to disk
# TODO read/write functions for storage
# TODO return full diagnostic information for downstream consumption
# TODO remove printer


class RobustNTF:
    """
    Class implementation of robust NTF functionality. See robust_ntf() for
    reference usage. It is preferred to use rntf.stats instead of rntf.obj.
    Downstream users are encouraged to use pandas to deal with the resulting
    statistics.
    """

    RANDOM = "random"
    USER = "user"

    FIT = "fit"
    REG = "regularization"
    OBJ = "objective"
    ERR = "error"
    LOG_ERR = "log_error"

    DATA_FILE = "data.pickle"
    DATA_BACKUP_FILE = "data.bak"
    CONFIG_FILE = "rntf_config.json"
    CONFIG_BACKUP_FILE = "rntf_config.bak"
    STATS_FILE = "stats.csv"
    STATS_BACKUP_FILE = "stats.bak"
    FILES = [
        DATA_FILE,
        DATA_BACKUP_FILE,
        CONFIG_FILE,
        CONFIG_BACKUP_FILE,
        STATS_FILE,
        STATS_BACKUP_FILE,
    ]

    def __init__(
        self,
        rank: int,
        beta: float,
        reg_val: float,
        tol: float,
        init: str = "random",
        max_iter: int = 10000,
        print_every: int = 100,
        user_prov: Union[dict, None] = None,
        save_every: int = 0,
        save_folder: Optional[PathLike] = None,
        printer: Callable[..., None] = print,
        allow_cpu: bool = False,
    ):
        rank = int(rank)
        assert 1 <= rank

        beta = float(beta)
        assert 0.0 <= beta <= 2.0

        reg_val = float(reg_val)
        assert 0.0 < reg_val

        tol = float(tol)
        assert 0.0 < tol

        init = str(init)
        assert init in [self.RANDOM, self.USER]

        max_iter = int(max_iter)
        assert 0 < max_iter

        print_every = int(print_every)
        assert 0 <= print_every

        save_every = int(save_every)
        assert 0 <= save_every

        if save_folder is not None:
            save_folder = PurePath(save_folder)
            assert Path(save_folder).is_dir()
            assert Path(save_folder).exists()

        self._save_folder = save_folder
        self._rank = rank
        self._beta = beta
        self._reg_val = reg_val
        self._tol = tol
        self._init = init
        self._max_iter = max_iter
        self._print_every = print_every
        self._user_prov = user_prov
        self._save_every = save_every
        self._printer = printer
        self._allow_cpu = allow_cpu

        self._stats = None
        self._matrices = None
        self._outlier = None
        self._data_n = None
        self._data_imputed = None
        self._data_approximation = None
        self._valid_mask = None
        self._eps = None
        self._device = None

    @property
    def max_iter(self) -> int:
        return self._max_iter

    @max_iter.setter
    def max_iter(self, value: int) -> None:
        value = int(value)
        assert 0 < value
        self._max_iter = value

    @property
    def matrices(self) -> List[torch.Tensor]:
        assert self._matrices is not None
        return [self._to_np(m) for m in self._matrices]

    @property
    def outlier(self) -> torch.Tensor:
        assert self._outlier is not None
        return self._to_np(self._outlier)

    @property
    def stats(self) -> pd.DataFrame:
        assert self._stats is not None
        return pd.DataFrame(self._stats)

    @property
    def obj(self) -> torch.Tensor:
        """
        For backward compatibility with previous interface
        """
        assert self._stats is not None
        assert self._device is not None
        obj = np.array([s[self.OBJ] for s in self._stats])
        obj = torch.from_numpy(obj).to(self._device)
        return obj

    @staticmethod
    def _to_np(data: torch.Tensor):
        return data.cpu().numpy()

    def apply(self, data: torch.Tensor):
        self._stats = None
        self._matrices = None
        self._outlier = None
        self._data_n = None
        self._data_imputed = None
        self._data_approximation = None
        self._valid_mask = None
        self._eps = None
        self._device = None

        self._initialize(data)

        assert self._matrices is not None
        assert self._outlier is not None
        assert self._data_n is not None
        assert self._data_imputed is not None
        assert self._data_approximation is not None
        assert self._valid_mask is not None
        assert self._eps is not None
        assert self._device is not None

        del data  # ! TODO this is dangerous!

        self._update_statistics()  # iteration = 0
        self._print_statistics(0)

        iteration = self._get_iteration()
        while self._do_continue(iteration):
            self._update_approximation()
            self._update_statistics()
            if self._do_print(iteration):
                self._print_statistics(iteration)
            if self._do_save(iteration):
                self.save()
            iteration += 1

    def _get_iteration(self) -> int:
        assert self._stats is not None
        return len(self._stats)

    def run(self):
        iteration = self._get_iteration()
        while self._do_continue(iteration):
            self._update_approximation()
            self._update_statistics()
            if self._do_print(iteration):
                self._print_statistics(iteration)
            if self._do_save(iteration):
                self.save()
            iteration += 1

    def _initialize(self, data: torch.Tensor):
        # Utilities:
        device = data.device
        is_cuda = "cuda" in device.type
        has_cuda = torch.cuda.is_available()

        if has_cuda and not self._allow_cpu and not is_cuda:
            print("GPU available, CPU not allowed. Moving data to GPU...")
            data = data.to("cuda:0")
        elif not has_cuda and not self._allow_cpu:
            raise RuntimeError("GPU not found, CPU not allowed. Stopping...")

        # Defining epsilon to protect against division by zero:
        if data.type() in ("torch.cuda.FloatTensor", "torch.FloatTensor"):
            eps = np.finfo(np.float32).eps
        else:
            eps = np.finfo(float).eps
        eps = eps * 1.1  # Slightly higher than actual epsilon

        # Initialize rNTF:
        matrices, outlier = initialize_rntf(
            data, self._rank, self._init, self._printer, self._user_prov
        )
        matrices = [m.to(device) for m in matrices]
        outlier = outlier.to(device)

        # Set up for the algorithm:
        # Initial approximation of the reconstruction:
        data_approximation = matrices[0] @ (kr_bcd(matrices, 0).t())
        data_approximation = data_approximation.to(device)
        data_approximation = folder(data_approximation, data, 0) + outlier + eps

        # EM step:
        valid_mask = torch.ones(data.size(), dtype=torch.bool)
        valid_mask[torch.isnan(data)] = False
        valid_mask = valid_mask.to(device)

        data_n = data.clone()
        data_n[torch.bitwise_not(valid_mask)] = 0.0

        data_imputed = self._impute_missing(
            data_approximation=data_approximation, valid_mask=valid_mask, data_n=data_n
        )

        self._matrices = matrices
        self._outlier = outlier
        self._data_n = data_n
        self._data_imputed = data_imputed
        self._data_approximation = data_approximation
        self._valid_mask = valid_mask
        self._eps = eps
        self._device = device

    def _update_approximation(self) -> None:
        assert self._data_n is not None
        assert self._data_approximation is not None
        assert self._valid_mask is not None

        # EM step:
        self._data_imputed = self._impute_missing(
            data_approximation=self._data_approximation,
            valid_mask=self._valid_mask,
            data_n=self._data_n,
        )

        # Block coordinate descent/loop through modes:
        modes = list(range(len(self._data_n.shape)))
        for mode in modes:

            # Khatri-Rao product of the matrices being held constant:
            kr_term = kr_bcd(self._matrices, mode).t()

            # Update factor matrix in mode of interest:
            self._matrices[mode] = update_factor(
                unfolder(self._data_imputed, mode),
                unfolder(self._data_approximation, mode),
                self._beta,
                self._matrices[mode],
                kr_term,
            )

            # Update reconstruction:
            self._data_approximation = (
                folder(self._matrices[mode] @ kr_term, self._data_n, mode)
                + self._outlier
                + self._eps
            )

            # Update outlier tensor:
            outlier = update_outlier(
                unfolder(self._data_imputed, mode),
                unfolder(self._data_approximation, mode),
                unfolder(self._outlier, mode),
                self._beta,
                self._reg_val,
            )
            self._outlier = folder(outlier, self._data_n, mode)

            # Update reconstruction:
            self._data_approximation = (
                folder(self._matrices[mode] @ kr_term, self._data_n, mode)
                + self._outlier
                + self._eps
            )

    def save(self, folder: Optional[PathLike] = None) -> None:
        if folder is None:
            assert self._save_folder is not None
            folder = self._save_folder
        folder = PurePath(folder)

        assert Path(folder).is_dir()
        assert Path(folder).exists()

        data = {
            "outlier": self._outlier,
            "data_n": self._data_n,
            "data_approximation": self._data_approximation,
            "valid_mask": self._valid_mask,
        }
        data = {k: v.cpu().numpy() for k, v in data.items()}
        data["matrices"] = [m.cpu().numpy() for m in self._matrices]
        data["eps"] = self._eps

        save_folder = self._save_folder
        if save_folder is not None:
            save_folder = str(save_folder)
        config = {
            "save_folder": save_folder,
            "device": str(self._device),
            "rank": self._rank,
            "beta": self._beta,
            "reg_val": self._reg_val,
            "tol": self._tol,
            "max_iter": self._max_iter,
            "print_every": self._print_every,
            "save_every": self._save_every,
            "allow_cpu": self._allow_cpu,
        }

        stats = self._stats
        stats = pd.DataFrame(stats)

        data_file = PurePath(folder / self.DATA_FILE)
        self._save_file(
            data=data, file_path=data_file, save_fn=pickle.dump, is_binary=True
        )
        config_file = PurePath(folder / self.CONFIG_FILE)
        self._save_file(data=config, file_path=config_file, save_fn=json.dump)
        stats_file = PurePath(folder / self.STATS_FILE)
        self._save_file(
            data=stats,
            file_path=stats_file,
            save_fn=lambda df, f: pd.DataFrame.to_csv(df, path_or_buf=f, index=False),
        )

    @staticmethod
    def _save_file(
        data: Any, file_path: PathLike, save_fn: Callable, is_binary: bool = False
    ):
        mode = "w"
        if is_binary:
            mode += "b"
        file_path = PurePath(file_path)
        backup_file_path = file_path.parent / (file_path.stem + ".bak")
        if Path(file_path).exists():
            Path(file_path).replace(backup_file_path)
        with open(file_path, mode) as f:
            save_fn(data, f)

    @staticmethod
    def _load_file(
        file_path: PathLike, load_fn: Callable, file_type: str, is_binary: bool = False
    ):
        mode = "r"
        if is_binary:
            mode += "b"
        file_path = PurePath(file_path)
        backup_file_path = file_path.parent / (file_path.stem + ".bak")

        out = None
        try:
            with open(PurePath(file_path), mode) as f:
                out = load_fn(f)
        except Exception as e:
            print("Unable to load {:s} file, trying backup.".format(file_type))
            print(e)
        if out is not None:
            return out

        try:
            with open(PurePath(backup_file_path), mode) as f:
                out = load_fn(f)
        except Exception as e:
            print("Unable to load {:s} file backup.".format(file_type))
            raise e
        return out

    @classmethod
    def load(cls, folder: PathLike, printer: Callable[..., None] = print):
        assert Path(folder).is_dir()
        assert Path(folder).exists()
        folder = PurePath(folder)

        data = cls._load_file(
            file_path=PurePath(folder / cls.DATA_FILE),
            load_fn=pickle.load,
            file_type="data",
            is_binary=True,
        )
        config = cls._load_file(
            file_path=PurePath(folder / cls.CONFIG_FILE),
            load_fn=json.load,
            file_type="rntf_config",
        )
        stats = cls._load_file(
            file_path=PurePath(folder / cls.STATS_FILE),
            load_fn=pd.read_csv,
            file_type="stats",
        )
        stats = stats.to_dict("records")

        device = torch.device(config["device"])
        out = cls(
            save_folder=config["save_folder"],
            rank=config["rank"],
            beta=config["beta"],
            reg_val=config["reg_val"],
            tol=config["tol"],
            max_iter=config["max_iter"],
            print_every=config["print_every"],
            save_every=config["save_every"],
            printer=printer,
            allow_cpu=config["allow_cpu"],
        )
        out._device = device

        matrices = data.pop("matrices")
        matrices = [torch.from_numpy(v).to(device) for v in matrices]
        eps = data.pop("eps")
        data = {k: torch.from_numpy(v).to(device) for k, v in data.items()}
        out._matrices = matrices
        out._eps = eps
        out._outlier = data["outlier"]
        out._data_n = data["data_n"]
        out._data_approximation = data["data_approximation"]
        out._valid_mask = data["valid_mask"]
        out._stats = stats
        return out

    def _do_continue(self, iteration) -> bool:
        do_continue = True
        if self._is_below_tolerance():
            self._print_below_tolerance()
            do_continue = False
        if self._is_enough_iterations(iteration):
            self._print_enough_iterations()
            do_continue = False
        return do_continue

    def _is_below_tolerance(self) -> bool:
        return self._get_last_err() <= self._tol

    def _print_below_tolerance(self) -> None:
        CONV_VALS = ", ".join(["{tol:.2e}", "log {log_tol:.4f}"])
        CONV_VALS = CONV_VALS.format(tol=self._tol, log_tol=self._get_log_tol())
        CONV_STATEMENT = "Algorithm converged per tolerance ({:s})"
        CONV_STATEMENT = CONV_STATEMENT.format(CONV_VALS)
        self._printer(CONV_STATEMENT)

    def _is_enough_iterations(self, iteration) -> bool:
        return self._max_iter <= iteration - 1

    def _print_enough_iterations(self) -> None:
        ITER_VALS = "{max_iter:d}"
        ITER_VALS = ITER_VALS.format(max_iter=self._max_iter)
        ITER_STATEMENT = "Maximum number of iterations achieved ({:s})"
        ITER_STATEMENT = ITER_STATEMENT.format(ITER_VALS)
        self._printer(ITER_STATEMENT)

    def _get_last_obj(self) -> float:
        assert self._stats is not None
        return self._get_last_stats()[self.OBJ]

    def _get_last_log_err(self) -> float:
        assert self._stats is not None
        return self._get_last_stats()[self.LOG_ERR]

    def _get_last_err(self) -> float:
        assert self._stats is not None
        return self._get_last_stats()[self.ERR]

    def _get_last_stats(self) -> Dict[str, Any]:
        assert self._stats is not None
        return self._stats[-1]

    def _get_log_tol(self) -> float:
        return np.log(self._tol).item()

    def _update_statistics(self) -> None:
        """
        if stats is None, initializes stats, err set to nan
        """
        fit = beta_divergence(self._data_imputed, self._data_approximation, self._beta)
        fit = fit.cpu().numpy().item()

        reg = self._reg_val * L21_norm(unfolder(self._outlier, 0))
        reg = reg.cpu().numpy().item()

        obj = fit + reg

        if self._stats is None:
            err = float("nan")
        else:
            prev_obj = self._get_last_obj()
            err = abs((prev_obj - obj) / prev_obj)

        log_err = np.log(err).item()

        entry = {
            self.FIT: fit,
            self.REG: reg,
            self.OBJ: obj,
            self.ERR: err,
            self.LOG_ERR: log_err,
        }
        if self._stats is None:
            self._stats = []
        self._stats.append(entry)

    def _print_statistics(self, iteration) -> None:
        STATEMENT = [
            "Iter: {iter: > 8d}",
            "Objective: {obj: > 6.4e}",
            "Log Error: {log_err: > 8.4f}",
            "Tol: {log_tol: > 8.4f}",
        ]
        STATEMENT = ", ".join(STATEMENT)
        formatted = STATEMENT.format(
            iter=iteration,
            obj=self._get_last_obj(),
            log_err=self._get_last_log_err(),
            log_tol=self._get_log_tol(),
        )
        self._printer(formatted)

    def _do_print(self, iteration: int) -> bool:
        if self._print_every == 0:
            out = False
        else:
            return iteration % self._print_every == 0
        return out

    def _do_save(self, iteration: int) -> bool:
        if self._save_every == 0:
            out = False
        else:
            out = iteration % self._save_every == 0
        return out

    @staticmethod
    def _impute_missing(
        data_approximation: torch.Tensor, valid_mask: torch.Tensor, data_n: torch.Tensor
    ) -> torch.Tensor:
        out = data_approximation.clone()
        out[valid_mask] = 0.0
        out += data_n
        return out


def robust_ntf(
    data: torch.Tensor,
    rank: int,
    beta: float,
    reg_val: float,
    tol: float,
    init: str = "random",
    max_iter: int = 10000,
    print_every: int = 100,
    printer: Callable[..., None] = print,
    user_prov: Union[dict, None] = None,
    allow_cpu: bool = False,
    save_every: int = 0,
    save_folder: Optional[PathLike] = None,
) -> Tuple[List[torch.Tensor], torch.Tensor, torch.Tensor]:
    """Robust Non-negative Tensor Factorization (rNTF)

    This function decomposes an input non-negative tensor into the sum of
    component rank-1 non-negative tensors (returned as a series of factor
    matrices), and a group-sparse non-negative outlier tensor that does not
    fit within a low-rank multi-linear model.

    The objective function is a weighted sum of the beta-divergence and L2,1
    norm, which allows for flexible noise modeling and imposing sparsity on the
    outliers. Missing values can be optionally handled via Expectation-
    Maximization. However, the model will no longer be identifiable.

    For more details, see Dey, N., et al. "Robust Non-negative Tensor
    Factorization, Diffeomorphic Motion Correction, and Functional Statistics
    to Understand Fixation in Fluorescence Microscopy", MICCAI, 2019.

    Parameters
    ----------
    data : tensor
        An n-dimensional non-negative tensor. Missing values should be NaNs.

    rank : int
        Rank of the factorization/number of components.

    beta : float, range [0, 2]
        Float parameterizing the beta divergence.
        Values at certain limits:
            beta = 2: Squared Euclidean distance (Gaussian noise assumption)
            beta = 1: Kullback-Leibler divergence (Poisson noise assumption)
            beta = 0: Itakura-Saito divergence (multiplicative gamma noise
            assumption)
        Float values in between these integers interpolate between assumptions.
        Values outside of this range contain dragons.

    reg_val : float
        Weight for the L2,1 penalty on the outlier tensor. Needs tuning
        specific to the range of the data. Start high and work your way down.

    tol : float
        tolerance on the iterative optimization.

    init : str, {'random' (default), 'user'}
        Initialization strategy.

        Valid options:
            1. 'random' : initialize all factor matrices and outlier tensor
                          uniformly at random.

            2. 'user' : you provide a dictionary containing initializations
                        for the factor matrices and outlier tensor. Must be
                        passed in the 'user_prov' paramter.

    max_iter : int
        Maximum number of iterations to compute rNTF.

    print_every : int
        Print optimization progress every 'print_every' iterations.

    printer : function
        Print-like function to receive progress updates.

    user_prov : None | dict
        Only relevant if init == 'user', i.e., you provide your own
        initialization. If so, provide a dictionary with the format:
        user_prov['factors'], user_prov['outlier'].


    Returns
    -------
    matrices : list
        A list of factor matrices retrieved from the decomposition.

    outlier : tensor
        The outlier tensor retrieved from the decomposition.

    obj : array, shape (n_iterations,)
        The history of the optimization.

    """
    assert max_iter > 0
    # TODO sanity checking

    rntf = RobustNTF(
        rank=rank,
        beta=beta,
        reg_val=reg_val,
        tol=tol,
        init=init,
        max_iter=max_iter,
        print_every=print_every,
        user_prov=user_prov,
        printer=printer,
        allow_cpu=allow_cpu,
        save_folder=save_folder,
        save_every=save_every,
    )

    rntf.apply(data)

    matrices = rntf.matrices
    outlier = rntf.outlier
    obj = rntf.obj

    return matrices, outlier, obj


def initialize_rntf(data, rank, alg, printer=print, user_prov=None):
    """Intialize Robust Non-negative Tensor Factorization.

    This function retrieves an initial estimate of factor matrices and an
    outlier tensor to intialize rNTF with.

    Parameters
    ----------
    data : matrix
        A matricized version of the input non-negative tensor.

    rank : int
        Rank of the factorization/number of components.

    alg : str, {'random' (default), 'user'}
        Initialization strategy.

        Valid options:
            1. 'random' : initialize all factor matrices and outlier tensor
                          uniformly at random.

            2. 'user' : you provide a dictionary containing initializations
                        for the factor matrices and outlier tensor. Must be
                        passed in the 'user_prov' parameter.

    printer : function
        Print-like function to receive status updates.

    user_prov : None | dict
        Only relevant if init == 'user', i.e., you provide your own
        initialization. If so, provide a dictionary with the format:
        user_prov['factors'], user_prov['outlier'].

    Returns
    -------
    matrices : list
        List of the initial factor matrices.

    outlier : tensor
        Intial estimate of the outlier tensor.

    """

    # Utilities:
    # Defining epsilon to protect against division by zero:
    if data.type() == "torch.cuda.FloatTensor":
        eps = 1.3e-7  # Slightly higher than actual epsilon in fp32
    else:
        eps = 2.3e-16  # Slightly higher than actual epsilon in fp64

    # Initialize outliers with uniform random values:
    outlier = torch.rand(data.size()) + eps

    # Initialize basis and coefficients:
    if alg == "random":
        printer("Initializing rNTF with uniform noise.")

        matrices = list()
        for idx in range(len(data.shape)):
            matrices.append(torch.rand(data.shape[idx], rank) + eps)

        return matrices, outlier

    elif alg == "user":
        printer("Initializing rNTF with user input.")

        matrices = user_prov["factors"]
        outlier = user_prov["outlier"]

        return matrices, outlier

    else:
        # Making sure the user doesn't do something unexpected:
        # Inspired by how sklearn deals with this:
        raise ValueError(
            "Invalid algorithm (typo?): got %r instead of one of %r"
            % (alg, ("random", "nndsvdar", "user"))
        )


def update_factor(data, data_approx, beta, factor, krp):
    """Update factor matrix.

    Implements the factor matrix update for robust non-negative tensor
    factorization.

    Parameters
    ----------
    data : matrix
        Matricized tensor in the mode corresponding to the factor matrix being
        solved for.

    data_approx : matrix
        Matricized version of the low-rank + sparse tensor reconstruction from
        the factor matrices, in the mode correspondidng to the factor matrix
        being solved for.

    beta : float, range [0, 2]
        Parameterization of the beta divergence. See docstring of robust_ntf()
        for details.

    factor : matrix
        Current estimate of the factor matrix being solved for.

    krp : matrix
        Current estimate of the Khatri-Rao product of the factor matrices
        currently being held constant while estimating 'factor', for block
        coordinate descent.

    Returns
    -------
    Multiplicative update for the factor matrix of interest.
    """

    return factor * (
        (data * (data_approx ** (beta - 2)) @ krp.t())
        / ((data_approx ** (beta - 1)) @ krp.t())
    )


def update_outlier(data, data_approx, outlier, beta, reg_val):
    """Update matricized outlier tensor.

    Implements the matricized outlier matrix update for robust non-negative
    tensor factorization.

    Parameters
    ----------
    data : matrix
        Matricized input tensor in the mode corresponding to the matricization
        of the outlier tensor.

    data_approx : matrix
        Matricized version of the low-rank tensor reconstruction from the
        factor matrices, in the mode correspondidng to the factor matrix being
        solved for.

    outlier : matrix
        Current estimate of the matricized outlier tensor being solved for.

    beta : float, range [0, 2]
        Parameterization of the beta divergence. See docstring of robust_ntf()
        for details.

    reg_val : float
        Weight for the L2,1 penalty on the outlier tensor. Needs tuning
        specific to the range of the data. Start high and work your way down.

    Returns
    -------
    Multiplicative update for the matricized outlier tensor.
    """

    # Utilities:
    # Defining epsilon to protect against division by zero:
    if data.type() == "torch.cuda.FloatTensor":
        eps = 1.3e-7  # Slightly higher than actual epsilon in fp32
    else:
        eps = 2.3e-16  # Slightly higher than actual epsilon in fp64

    # Using inline functions for readability:
    bet1 = lambda X: X ** (beta - 1)
    bet2 = lambda X: X ** (beta - 2)

    return outlier * (
        (data * bet2(data_approx))
        / (bet1(data_approx) + reg_val * normalize(outlier, p=2, dim=0, eps=eps))
    )
