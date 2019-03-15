

import types

import numpy as np
import unittest

import chainer
from chainer import optimizers

from chainer_profutil import create_marked_profile_optimizer


class TestCreateMarkedProfileOptimizer(unittest.TestCase):
    def test_creation_class_generate_func(self):
        basecls = optimizers.SGD
        sync = True

        generate_func = create_marked_profile_optimizer(
            basecls, sync=sync)
        self.assertIsNotNone(generate_func)
        self.assertIsInstance(generate_func, types.FunctionType)

    def test_creation_newsubclass_instance(self):
        basecls = optimizers.SGD
        sync = True

        generate_func = create_marked_profile_optimizer(
            basecls, sync=sync)
        optimizer = generate_func()
        self.assertIsNotNone(optimizer)
        self.assertTrue(issubclass(optimizer.__class__, chainer.optimizer.Optimizer))
        self.assertIsInstance(optimizer, chainer.Optimizer)

    def test_initialize_instance(self):
        basecls = optimizers.SGD
        sync = True

        generate_func = create_marked_profile_optimizer(
            basecls, sync=sync)
        optimizer = generate_func(lr=1.0)
        self.assertIsNotNone(optimizer)
        np.testing.assert_allclose([optimizer.lr], [1.0])

    def test_can_create_different_instances(self):
        basecls = optimizers.SGD
        sync = True

        generate_func = create_marked_profile_optimizer(
            basecls, sync=sync)

        optimizer1 = generate_func()
        optimizer2 = generate_func()
        opts = [optimizer1, optimizer2]
        for o in opts:
            self.assertIsNotNone(o)
            self.assertTrue(issubclass(o.__class__, chainer.Optimizer))
            self.assertIsInstance(o, chainer.Optimizer)
        self.assertIsNot(opts[0], opts[1])


class TestCreateMarkedProfileOptimizerError(unittest.TestCase):
    def test_fail_on_none_basecls(self):
        basecls = None
        sync = True

        with self.assertRaises(AssertionError):
            create_marked_profile_optimizer(
                basecls, sync=sync)

    def test_fail_on_unexpected_basecls(self):
        class DummyClass(object):
            def __init__(self):
                pass

        basecls = DummyClass
        sync = True

        with self.assertRaises(RuntimeError):
            create_marked_profile_optimizer(
                basecls, sync=sync)
