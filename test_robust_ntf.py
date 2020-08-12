from pathlib import Path, PurePath
import unittest
import atexit
import uuid

import torch
import numpy as np
import pandas as pd
import tensorly as tl
import matplotlib.pyplot as plt
import scipy
import scipy.signal
from tensorly.kruskal_tensor import kruskal_to_tensor
from tensorly.tenalg.outer_product import outer

from robust_ntf.robust_ntf import robust_ntf, RobustNTF

torch.set_default_tensor_type(torch.cuda.DoubleTensor)
tl.set_backend("pytorch")


class TestRobustNTF(unittest.TestCase):
    def setUp(self):
        torch.manual_seed(32)
        np.random.seed(32)
        EPS = 1e-6

        x_support = np.linspace(-1, 1, 50, endpoint=False)
        x1, x2, x3 = scipy.signal.gausspulse(x_support, fc=5, retquad=True, retenv=True)
        X = np.array([x1, x2, x3])
        X = torch.from_numpy(X)
        X = X - X.min() + EPS

        y_support = np.linspace(-1, 1, 96, endpoint=False)
        y1 = scipy.signal.chirp(y_support, f0=4, t1=-0.5, f1=4)
        y2 = scipy.signal.chirp(y_support, f0=2, t1=0.5, f1=3)
        y3 = scipy.signal.chirp(y_support, f0=1, t1=0.1, f1=2)
        Y = np.array([y1, y2, y3])
        Y = torch.from_numpy(Y)
        Y = Y - Y.min() + EPS

        z_support = np.linspace(0, 10, 20)
        z1 = scipy.stats.gamma(7).pdf(z_support)
        z2 = scipy.stats.gamma(4).pdf(z_support)
        z3 = scipy.stats.gamma(2).pdf(z_support)
        Z = np.array([z1, z2, z3])
        Z = torch.from_numpy(Z)
        Z = Z + EPS

        t = (None, [X.t(), Y.t(), Z.t()])
        t = kruskal_to_tensor(t)
        self.t = t

    def test_save_load_idempotency(self):
        SAVE_FOLDER = PurePath(str(uuid.uuid4()))
        Path(SAVE_FOLDER).mkdir(parents=False, exist_ok=False)
        rntf = RobustNTF(
            rank=3,
            beta=2.0,
            reg_val=0.1,
            tol=1e-8,
            max_iter=1,
            print_every=100,
            save_every=25,
            save_folder=SAVE_FOLDER,
        )
        rntf.apply(self.t.cuda()),
        save_points = np.random.randint(2, 1000, [50])
        save_points = sorted(list(save_points))
        atexit.register(lambda x: Path(x).rmdir(), SAVE_FOLDER)
        paths = [PurePath(SAVE_FOLDER) / f for f in RobustNTF.FILES]
        for path in paths:
            atexit.register(
                lambda x: Path(x).unlink(), path
            )  # change to missing_ok=True in 3.8
        for iteration in save_points:
            rntf.save()
            rntf_loaded = RobustNTF.load(SAVE_FOLDER)
            actual = rntf_loaded
            expected = rntf
            self.assertEqual(actual.max_iter, expected.max_iter)
            np.testing.assert_array_equal(actual.outlier, expected.outlier)
            for a, e in zip(actual.matrices, expected.matrices):
                np.testing.assert_array_equal(a, e)
            pd.testing.assert_frame_equal(actual.stats, expected.stats)

            rntf = rntf_loaded
            rntf.max_iter = iteration
            rntf.run()

    def test_iterations(self):
        factors, outlier, objective = robust_ntf(
            data=self.t.cuda(),
            rank=3,
            beta=2.0,
            reg_val=0.1,
            tol=1e-2,
            max_iter=1,
            print_every=1,
        )
        actual = torch.numel(objective)
        expected = 2  # index 0 is pre-iter, so max_iter=1 --> iters==2
        self.assertEqual(torch.numel(objective), expected)
