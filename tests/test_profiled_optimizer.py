

import types

import numpy as np
import unittest

import chainer
import chainer.links as L
from chainer import optimizers

import chainermn

from chainer_profutil import create_marked_profile_optimizer
from chainer_profutil.profiled_optimizer import _MarkedProfileOptimizer
from chainer_profutil.profiled_optimizer import _MarkedProfileOptimizerForMN


class TestCreateMarkedProfileOptimizer(unittest.TestCase):
    def test_initialize_instance(self):
        optimizer = create_marked_profile_optimizer(
            optimizers.SGD(lr=1.0), sync=True)
        self.assertIsNotNone(optimizer)
        np.testing.assert_allclose([optimizer.lr], [1.0])

    def test_can_create_different_instances(self):
        optimizer1 = create_marked_profile_optimizer(
            optimizers.SGD(), sync=True)
        optimizer2 = create_marked_profile_optimizer(
            optimizers.SGD(), sync=True)
        opts = [optimizer1, optimizer2]
        for o in opts:
            self.assertIsNotNone(o)
        self.assertIsNot(opts[0], opts[1])

    def test_can_create_valid_wrapper(self):
        optimizer = create_marked_profile_optimizer(
            optimizers.SGD(lr=1.0), sync=True)
        self.assertIsNotNone(optimizer)
        np.testing.assert_allclose([optimizer.lr], [1.0])
        self.assertIsInstance(
            optimizer,
            _MarkedProfileOptimizer)
        self.assertIsInstance(
            optimizer.actual_optimizer,
            chainer.Optimizer)

    def test_can_create_valid_wrapper_for_chainermn(self):
        optimizer = create_marked_profile_optimizer(
            chainermn.create_multi_node_optimizer(optimizers.SGD(lr=1.0), None),
            sync=True)
        self.assertIsNotNone(optimizer)
        np.testing.assert_allclose([optimizer.lr], [1.0])
        self.assertIsInstance(
            optimizer,
            _MarkedProfileOptimizerForMN)
        self.assertNotIsInstance(
            optimizer.actual_optimizer,
            chainer.Optimizer)
        self.assertIsInstance(
            optimizer.actual_optimizer.actual_optimizer,
            chainer.Optimizer)


class TestCreateMarkedProfileOptimizerError(unittest.TestCase):
    def test_fail_on_none_instance(self):
        with self.assertRaises(AssertionError):
            create_marked_profile_optimizer(
                None, sync=True)

    def test_fail_on_invalid_link(self):
        class InvalidNetwork(chainer.Chain):
            def __init__(self):
                super(InvalidNetwork, self).__init__()
                with self.init_scope():
                    self.l1 = L.Linear(None, 100)
                    self.l2 = L.Linear(None, 10)

            def __call__(self, x):
                h1 = F.relu(self.l1(x))
                return self.l2(h1)

        optimizer = create_marked_profile_optimizer(
            optimizers.SGD(), sync=True)
        with self.assertRaises(RuntimeError):
            optimizer.setup(InvalidNetwork())
