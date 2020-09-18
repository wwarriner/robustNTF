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

# TODO test cases
# 1) valueerror thrown on constant-valued input
# 2) check GPU device handling (i.e. missing)
# 3) check dtypes of outputs are consistent with input (np.float32, np.float64)
# 4) add check that input is FP and throw if not, check with test case
# 5) move main API entrypoints to top of file, move private-ish stuff below


import json
from pathlib import Path, PurePath
import pickle
from typing import Any, Callable, Dict, List, Optional, Tuple, Union

import numpy as np
import pandas as pd
import tensorly as tl
import torch
from torch.nn.functional import normalize
from tensorly.tenalg.outer_product import outer

from .foldings import folder, unfolder
from .matrix_utils import L21_norm, beta_divergence, kr_bcd

tl.set_backend("pytorch")

PathLike = Union[str, Path, PurePath]


class RntfConfig:
    RANDOM = "random"
    USER = "user"

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

        self.rank = rank
        self.beta = beta
        self.reg_val = reg_val
        self.tol = tol
        self.init = init
        self._max_iter = max_iter
        self.print_every = print_every
        self.save_every = save_every
        self.save_folder = save_folder
        self.allow_cpu = allow_cpu
        self.user_prov = user_prov

    def should_print(self, iteration: int) -> bool:
        if self.print_every == 0:
            out = False
        else:
            return iteration % self.print_every == 0
        return out

    def should_save(self, iteration: int) -> bool:
        if self.save_every == 0:
            out = False
        else:
            out = iteration % self.save_every == 0
        return out

    @property
    def max_iter(self) -> int:
        return self._max_iter

    @max_iter.setter
    def max_iter(self, value: int) -> None:
        value = int(value)
        assert 0 < value
        self._max_iter = value

    @property
    def log_tol(self) -> float:
        return np.log(self.tol).item()

    def save(self, file_path: PathLike) -> None:
        save_folder = self.save_folder
        if save_folder is not None:
            save_folder = str(save_folder)
        config = {
            "save_folder": save_folder,
            "rank": self.rank,
            "beta": self.beta,
            "reg_val": self.reg_val,
            "tol": self.tol,
            "max_iter": self.max_iter,
            "print_every": self.print_every,
            "save_every": self.save_every,
            "allow_cpu": self.allow_cpu,
        }
        _save_file(data=config, file_path=file_path, save_fn=json.dump)

    @classmethod
    def load(cls, file_path: PathLike) -> "RntfConfig":
        config = _load_file(
            file_path=file_path, load_fn=json.load, file_type="rntf_config"
        )
        return cls(**config)


class RobustNTF:
    """
    Class implementation of robust NTF functionality. See robust_ntf() for
    reference usage. It is preferred to use rntf.stats instead of rntf.obj.
    Downstream users are encouraged to use pandas to deal with the resulting
    statistics.
    """

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
    PREFERRED_FILES = [DATA_FILE, CONFIG_FILE, STATS_FILE]

    def __init__(self, config: RntfConfig):
        self._config = config
        self._data = None
        self._stats = RntfStats(config)

    @property
    def matrices(self) -> List[torch.Tensor]:
        assert self._data is not None
        assert self._data.ready
        return [m for m in self._data.matrices]

    @property
    def outlier(self) -> torch.Tensor:
        assert self._data is not None
        assert self._data.outlier is not None
        return self._data.outlier

    @property
    def stats(self) -> pd.DataFrame:
        return self._stats.to_df()

    @property
    def reconstruction(self) -> torch.Tensor:
        assert self._data is not None
        assert self._data.ready
        return self._data.reconstruct()

    @property
    def reconstructions_per_mode(self) -> List[torch.Tensor]:
        assert self._data is not None
        assert self._data.ready
        return [self._data.reconstruct_mode(rank) for rank in range(self._config.rank)]

    @property
    def valid_mask(self) -> torch.Tensor:
        assert self._data is not None
        assert self._data.valid_mask is not None
        return self._data.valid_mask

    @property
    def obj(self) -> torch.Tensor:
        """
        For backward compatibility with previous interface
        """
        return self._stats.obj

    def run(self, initial_data: Optional[torch.Tensor] = None) -> None:
        """
        If t is None, will attempt to continue from where it left off
        """
        if initial_data is not None:
            data = RntfData(config=self._config)
            data.initialize(data=initial_data)
            self._data = data

            del initial_data  # ! TODO this is dangerous!
            self._update_stats(iteration=0)  # iteration = 0

        iteration = self._stats.iteration
        while self._do_continue(iteration):
            self._data.update()
            self._update_stats(iteration=iteration)
            if self._config.should_save(iteration):
                self.save()
            iteration += 1

    def _update_stats(self, iteration: int) -> None:
        fit = self._data.compute_beta_divergence()
        reg = self._data.compute_regularization_term()
        L2 = self._data.compute_L2_accuracy()
        Linf = self._data.compute_Linf_accuracy()
        self._stats.update(fit=fit, reg=reg, L2=L2, Linf=Linf)
        if self._config.should_print(iteration):
            print(self._stats.iteration_to_string(iteration))

    def save(self, folder: Optional[PathLike] = None) -> None:
        if folder is None:
            assert self._config.save_folder is not None
            folder = self._config.save_folder
        folder = PurePath(folder)

        assert Path(folder).is_dir()
        assert Path(folder).exists()

        config_file = PurePath(folder / self.CONFIG_FILE)
        self._config.save(file_path=config_file)
        data_file = PurePath(folder / self.DATA_FILE)
        self._data.save(file_path=data_file)
        stats_file = PurePath(folder / self.STATS_FILE)
        self._stats.save(file_path=stats_file)

    @classmethod
    def check_data_exists(cls, folder: PathLike) -> bool:
        ok = True

        folder = PurePath(folder)
        ok = ok and Path(folder).is_dir()
        ok = ok and Path(folder).exists()

        config_file = PurePath(folder / cls.CONFIG_FILE)
        ok = ok and _check_file(config_file)

        data_file = PurePath(folder / cls.DATA_FILE)
        ok = ok and _check_file(data_file)

        stats_file = PurePath(folder / cls.STATS_FILE)
        ok = ok and _check_file(stats_file)

        return ok

    @classmethod
    def load(cls, folder: PathLike) -> Tuple["RobustNTF", "RntfConfig"]:
        assert cls.check_data_exists(folder)

        config_file = PurePath(folder / cls.CONFIG_FILE)
        config = RntfConfig.load(config_file)
        data_file = PurePath(folder / cls.DATA_FILE)
        data = RntfData.load(data_file, config=config)
        stats_file = PurePath(folder / cls.STATS_FILE)
        stats = RntfStats.load(stats_file, config=config)

        out = cls(config=config)
        out._data = data
        out._stats = stats
        return out, config

    def _do_continue(self, iteration) -> bool:
        do_continue = True
        if self._is_below_tolerance():
            self._print_final(iteration)
            self._print_below_tolerance()
            do_continue = False
        if self._is_enough_iterations(iteration):
            self._print_final(iteration)
            self._print_enough_iterations()
            do_continue = False
        return do_continue

    def _print_final(self, iteration) -> None:
        if not self._config.should_print(iteration):
            print(self._stats.iteration_to_string(iteration))

    def _is_below_tolerance(self) -> bool:
        return self._stats.err <= self._config.tol

    def _print_below_tolerance(self) -> None:
        CONV_VALS = ", ".join(["{tol:.2e}", "log {log_tol:.4f}"])
        CONV_VALS = CONV_VALS.format(tol=self._config.tol, log_tol=self._config.log_tol)
        CONV_STATEMENT = "Algorithm converged per tolerance ({:s})"
        CONV_STATEMENT = CONV_STATEMENT.format(CONV_VALS)
        print(CONV_STATEMENT)

    def _is_enough_iterations(self, iteration) -> bool:
        return self._config.max_iter <= iteration - 1

    def _print_enough_iterations(self) -> None:
        ITER_VALS = "{max_iter:d}"
        ITER_VALS = ITER_VALS.format(max_iter=self._config.max_iter)
        ITER_STATEMENT = "Maximum number of iterations achieved ({:s})"
        ITER_STATEMENT = ITER_STATEMENT.format(ITER_VALS)
        print(ITER_STATEMENT)


def robust_ntf(
    data: torch.Tensor,
    rank: int,
    beta: float,
    reg_val: float,
    tol: float,
    init: str = "random",
    max_iter: int = 10000,
    print_every: int = 100,
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
    config = RntfConfig(
        rank=rank,
        beta=beta,
        reg_val=reg_val,
        tol=tol,
        init=init,
        max_iter=max_iter,
        print_every=print_every,
        user_prov=user_prov,
        allow_cpu=allow_cpu,
        save_every=save_every,
        save_folder=save_folder,
    )
    rntf = RobustNTF(config)
    rntf.run(data)

    matrices = rntf.matrices
    outlier = rntf.outlier
    obj = rntf.obj

    return matrices, outlier, obj


def initialize_rntf(data, rank, alg, user_prov=None):
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
        print("Initializing rNTF with uniform noise.")

        matrices = list()
        for idx in range(len(data.shape)):
            matrices.append(torch.rand(data.shape[idx], rank) + eps)

        return matrices, outlier

    elif alg == "user":
        print("Initializing rNTF with user input.")

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


class RntfData:
    def __init__(self, config: RntfConfig):
        self._config = config
        self.matrices = None
        self.outlier = None
        self.data_n = None
        self.data_imputed = None
        self.data_approximation = None
        self.valid_mask = None
        self._eps = None
        self._device = None

    @property
    def ready(self) -> bool:
        ok = True
        ok = ok and self.matrices is not None
        ok = ok and self.outlier is not None
        ok = ok and self.data_n is not None
        ok = ok and self.data_approximation is not None
        ok = ok and self.valid_mask is not None
        ok = ok and self._eps is not None
        ok = ok and self._device is not None
        return ok

    def initialize(self, data: torch.Tensor) -> None:
        # Utilities:
        device = data.device
        is_cuda = "cuda" in device.type
        has_cuda = torch.cuda.is_available()

        if has_cuda and not self._config.allow_cpu and not is_cuda:
            print("GPU available, CPU not allowed. Moving data to GPU...")
            data = data.to("cuda:0")
        elif not has_cuda and not self._config.allow_cpu:
            raise RuntimeError("GPU not found, CPU not allowed. Stopping...")

        # Defining epsilon to protect against division by zero:
        if data.type() in ("torch.cuda.FloatTensor", "torch.FloatTensor"):
            eps = np.finfo(np.float32).eps
        else:
            eps = np.finfo(float).eps
        eps = eps * 1.1  # Slightly higher than actual epsilon

        # Initialize rNTF:
        matrices, outlier = initialize_rntf(
            data, self._config.rank, self._config.init, self._config.user_prov
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

        self._device = device
        self._eps = eps
        self.matrices = matrices
        self.outlier = outlier
        self.data_n = data_n
        self.data_approximation = data_approximation
        self.valid_mask = valid_mask
        self.data_imputed = data_imputed

    @property
    def mode_count(self) -> int:
        return len(self.data_n.shape)

    @property
    def modes(self) -> List[int]:
        return list(range(self.mode_count))

    @property
    def shape(self) -> List[int]:
        assert self.data_n is not None
        return list(self.data_n.shape)

    @property
    def ndim(self) -> int:
        return len(self.shape)

    def update(self) -> None:
        assert self.ready
        assert self.data_approximation is not None
        assert self.valid_mask is not None
        assert self.data_n is not None

        # EM step:
        self.data_imputed = self._impute_missing(
            data_approximation=self.data_approximation,
            valid_mask=self.valid_mask,
            data_n=self.data_n,
        )

        # Block coordinate descent/loop through modes:
        for mode in self.modes:

            # Khatri-Rao product of the matrices being held constant:
            kr_term = kr_bcd(self.matrices, mode).t()

            # Update factor matrix in mode of interest:
            self.matrices[mode] = update_factor(
                unfolder(self.data_imputed, mode),
                unfolder(self.data_approximation, mode),
                self._config.beta,
                self.matrices[mode],
                kr_term,
            )

            # Update reconstruction:
            self.data_approximation = (
                folder(self.matrices[mode] @ kr_term, self.data_n, mode)
                + self.outlier
                + self._eps
            )

            # Update outlier tensor:
            outlier = update_outlier(
                unfolder(self.data_imputed, mode),
                unfolder(self.data_approximation, mode),
                unfolder(self.outlier, mode),
                self._config.beta,
                self._config.reg_val,
            )
            self.outlier = folder(outlier, self.data_n, mode)

            # Update reconstruction:
            self.data_approximation = (
                folder(self.matrices[mode] @ kr_term, self.data_n, mode)
                + self.outlier
                + self._eps
            )

    def compute_beta_divergence(self) -> float:
        assert self.data_imputed is not None
        assert self.data_approximation is not None
        assert self.data_approximation is not None
        out = beta_divergence(
            self.data_imputed, self.data_approximation, self._config.beta
        )
        out = out.cpu().numpy().item()
        return out

    def compute_regularization_term(self) -> float:
        assert self.outlier is not None
        out = L21_norm(unfolder(self.outlier, 0))
        out = self._config.reg_val * out
        out = out.cpu().numpy().item()
        return out

    def compute_L2_accuracy(self) -> float:
        assert self.data_approximation is not None
        assert self.data_n is not None
        assert self.valid_mask is not None
        out = self._compute_error(self.data_approximation, self.data_n)
        out = out[self.valid_mask]
        out = ((out ** 2).sum()) ** 0.5
        out = out.item()
        assert isinstance(out, float)
        return out

    def compute_Linf_accuracy(self) -> float:
        assert self.data_approximation is not None
        assert self.data_n is not None
        assert self.valid_mask is not None
        out = self._compute_error(self.data_approximation, self.data_n)
        out = out[self.valid_mask]
        out = out.max()
        out = out.item()
        assert isinstance(out, float)
        return out

    def save(self, file_path: PathLike) -> None:
        assert self.ready
        data = {
            "outlier": self.outlier,
            "data_n": self.data_n,
            "data_approximation": self.data_approximation,
            "valid_mask": self.valid_mask,
        }
        data = {k: v.cpu().numpy() for k, v in data.items()}

        data["matrices"] = [m.cpu().numpy() for m in self.matrices]
        data["eps"] = self._eps
        data["device"] = str(self._device)

        _save_file(data=data, file_path=file_path, save_fn=pickle.dump, is_binary=True)

    @classmethod
    def load(cls, file_path: PathLike, config: RntfConfig) -> "RntfData":
        data = _load_file(
            file_path=file_path, load_fn=pickle.load, file_type="data", is_binary=True
        )
        device = data.pop("device")
        eps = data.pop("eps")

        matrices = data.pop("matrices")
        matrices = [torch.from_numpy(v).to(device) for v in matrices]

        data = {k: torch.from_numpy(v).to(device) for k, v in data.items()}

        data_approximation = data["data_approximation"]
        valid_mask = data["valid_mask"]
        data_n = data["data_n"]
        data_imputed = cls._impute_missing(
            data_approximation=data_approximation, valid_mask=valid_mask, data_n=data_n
        )

        out = cls(config=config)
        out._device = device
        out._eps = eps
        out.matrices = matrices
        out.outlier = data["outlier"]
        out.data_n = data_n
        out.data_approximation = data_approximation
        out.valid_mask = valid_mask
        out.data_imputed = data_imputed
        return out

    def reconstruct(self) -> torch.Tensor:
        out = torch.zeros(self.shape, device=self._device)
        for rank in list(range(self._config.rank)):
            out += self.reconstruct_mode(rank)
        return out

    def reconstruct_mode(self, rank: int) -> torch.Tensor:
        factors = [self.matrices[d][:, rank] for d in range(self.ndim)]
        out = outer(factors)
        return out

    @staticmethod
    def _compute_error(lhs: torch.Tensor, rhs: torch.Tensor) -> torch.Tensor:
        return torch.abs(lhs - rhs)

    @staticmethod
    def _impute_missing(
        data_approximation: torch.Tensor, valid_mask: torch.Tensor, data_n: torch.Tensor
    ) -> torch.Tensor:
        out = data_approximation.clone()
        out[valid_mask] = 0.0
        out += data_n
        return out


class RntfStats:
    FIT = "fit"
    REG = "regularization_term"
    OBJ = "objective"
    ERR = "error"
    LOG_ERR = "log_error"
    L2_ACC = "L2_accuracy"
    LINF_ACC = "Linf_accuracy"

    def __init__(self, config: RntfConfig):
        self._config = config
        self._stats = None

    @property
    def obj(self) -> torch.Tensor:
        """
        For backward compatibility with previous interface
        """
        assert self._stats is not None
        obj = np.array([s[self.OBJ] for s in self._stats])
        obj = torch.from_numpy(obj)
        return obj

    @property
    def iteration(self) -> int:
        assert self._stats is not None
        return len(self._stats)  # iteration 0 is special for initialized state

    @property
    def err(self) -> float:
        assert self._stats is not None
        return self._get_last_err()

    def update(self, fit: float, reg: float, L2: float, Linf: float) -> None:
        """
        if stats is None, initializes stats, err set to nan
        """
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
            self.L2_ACC: L2,
            self.LINF_ACC: Linf,
        }
        if self._stats is None:
            self._stats = []
        self._stats.append(entry)

    def iteration_to_string(self, iteration: int) -> str:
        STATEMENT = [
            "Iter: {iter: > 8d}",
            "Objective: {obj: > 6.4e}",
            "Log Error: {log_err: > 8.4f}",
            "Tol: {log_tol: > 8.4f}",
            # TODO accuracy
        ]
        STATEMENT = ", ".join(STATEMENT)
        formatted = STATEMENT.format(
            iter=iteration,
            obj=self._get_last_obj(),
            log_err=self._get_last_log_err(),
            log_tol=self._get_log_tol(),
        )
        return formatted

    def to_df(self) -> pd.DataFrame:
        return pd.DataFrame(self._stats)

    def save(self, file_path: PathLike) -> None:
        _save_file(
            data=self.to_df(),
            file_path=file_path,
            save_fn=lambda df, f: pd.DataFrame.to_csv(
                df, path_or_buf=f, index=False, line_terminator="\n"
            ),
        )

    @classmethod
    def load(cls, file_path: PathLike, config: RntfConfig) -> "RntfStats":
        stats = _load_file(file_path=file_path, load_fn=pd.read_csv, file_type="stats")
        stats = stats.to_dict("records")

        out = cls(config=config)
        out._stats = stats
        return out

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
        return np.log(self._config.tol).item()


def _save_file(
    data: Any, file_path: PathLike, save_fn: Callable, is_binary: bool = False
) -> None:
    mode = "w"
    if is_binary:
        mode += "b"
    file_path = PurePath(file_path)
    backup_file_path = _to_backup_file(file_path)
    if Path(file_path).exists():
        Path(file_path).replace(backup_file_path)
    with open(file_path, mode) as f:
        save_fn(data, f)


def _load_file(
    file_path: PathLike, load_fn: Callable, file_type: str, is_binary: bool = False
) -> Any:
    mode = "r"
    if is_binary:
        mode += "b"
    file_path = PurePath(file_path)
    backup_file_path = _to_backup_file(file_path)

    out = None
    try:
        with open(PurePath(file_path), mode) as f:
            out = load_fn(f)
    except Exception as e:
        print("Unable to load {:s} file, trying backup.".format(file_type))
    if out is not None:
        return out

    try:
        with open(PurePath(backup_file_path), mode) as f:
            out = load_fn(f)
    except Exception as e:
        print("Unable to load {:s} file backup.".format(file_type))
        raise e
    return out


def _check_file(file_path: PathLike) -> bool:
    file_path = PurePath(file_path)
    backup_file_path = _to_backup_file(file_path)

    ok = True
    ok = ok and Path(file_path).is_file()
    ok = ok and Path(file_path).exists()

    if ok:
        return ok

    ok = True
    ok = ok and Path(backup_file_path).is_file()
    ok = ok and Path(backup_file_path).exists()

    return ok


def _to_backup_file(file_path: PathLike) -> PurePath:
    file_path = PurePath(file_path)
    return file_path.parent / (file_path.stem + ".bak")
