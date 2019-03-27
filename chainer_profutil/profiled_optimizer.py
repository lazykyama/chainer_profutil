

from chainer.optimizer import Optimizer
from chainer.function_hooks import CUDAProfileHook

from cupy import cuda
from cupy import prof
from cupy.cuda import runtime


_itr_argb_color = 0xfffff100
_fwd_argb_color = 0xff76b900
_bwd_argb_color = 0xff7fdbff
_upd_argb_color = 0xffe60012


def _try_to_sync_if_needed(sync):
    if sync:
        runtime.deviceSynchronize()

def range_push(sync, msg, argb_color):
    _try_to_sync_if_needed(sync)
    if argb_color is None:
        cuda.nvtx.RangePush(msg)
    else:
        cuda.nvtx.RangePushC(msg, argb_color)

def range_pop(sync):
    _try_to_sync_if_needed(sync)
    cuda.nvtx.RangePop()


class UpdateProfileMarkPreHook(object):
    name = 'preupdate'
    call_for_each_param = False
    timing = 'pre'

    def __init__(self, sync=True, argb_color=None):
        self._sync = sync
        self._argb_color = argb_color

    def __call__(self, rule):
        range_push(self._sync, 'model.update', self._argb_color)

class UpdateProfileMarkPostHook(object):
    name = 'postupdate'
    call_for_each_param = False
    timing = 'post'

    def __init__(self, sync=True, seprately_mark_for_iter=False):
        self._sync = sync
        self._seprately_mark_for_iter = seprately_mark_for_iter

    def __call__(self, rule):
        range_pop(self._sync)  # pop 'model.update'
        if self._seprately_mark_for_iter:
            range_pop(self._sync)  # pop 'iteration'


class FwdBwdProfileMarkHook(CUDAProfileHook):

    name = 'FwdBwdProfileMarkHook'

    def __init__(self, sync=True, argb_color=None):
        super(FwdBwdProfileMarkHook, self).__init__()
        self._sync = sync
        self._argb_color = argb_color

    def forward_preprocess(self, function, in_data):
        range_push(self._sync,
                   function.label + '.forward',
                   self._argb_color)

    def forward_postprocess(self, function, in_data):
        range_pop(self._sync)

    def backward_preprocess(self, function, in_data, out_grad):
        range_push(self._sync,
                   function.label + '.backward',
                   self._argb_color)

    def backward_postprocess(self, function, in_data, out_grad):
        range_pop(self._sync)


def _add_backward_mark(func, sync, layerwise_sync):
    def backward_wrapper(*args, **kwargs):
        with prof.TimeRangeDecorator('model.backward', sync=sync, argb_color=_bwd_argb_color):
            with FwdBwdProfileMarkHook(sync=layerwise_sync, argb_color=_bwd_argb_color):
                ret = func(*args, **kwargs)
        return ret
    return backward_wrapper


def make_wrapped_lossfunc(func,
                          sync=True,
                          layerwise_sync=False,
                          seprately_mark_for_iter=True):
    if func is None:
        raise ValueError('func is required.')
    if not hasattr(func, 'forward'):
        raise RuntimeError('func must have forward method.')

    def forward_wrapper(*args, **kwargs):
        if seprately_mark_for_iter:
            range_push(sync, 'iteration', _itr_argb_color)

        with prof.TimeRangeDecorator('model.forward', sync=sync, argb_color=_fwd_argb_color):
            with FwdBwdProfileMarkHook(sync=layerwise_sync, argb_color=_fwd_argb_color):
                ret = func._org_forward(*args, **kwargs)
        ret.backward = _add_backward_mark(ret.backward, sync, layerwise_sync)
        return ret

    func._org_forward = func.forward
    func.forward = forward_wrapper
    return func


class _MarkedProfileOptimizerBase(object):
    def __init__(self, actual_optimizer, sync=False, layerwise_sync=False):
        super(_MarkedProfileOptimizerBase, self).__setattr__(
            'actual_optimizer', actual_optimizer)
        super(_MarkedProfileOptimizerBase, self).__setattr__(
            '_sync', sync)
        super(_MarkedProfileOptimizerBase, self).__setattr__(
            '_layerwise_sync', layerwise_sync)

    def _setup(self, link, seprately_mark_for_iter=True):
        make_wrapped_lossfunc(
            link,
            sync=self._sync,
            layerwise_sync=self._layerwise_sync,
            seprately_mark_for_iter=seprately_mark_for_iter)
        ret = self.actual_optimizer.setup(link)

        self.actual_optimizer.add_hook(
            UpdateProfileMarkPreHook(sync=self._sync,
                                     argb_color=_upd_argb_color))
        self.actual_optimizer.add_hook(
            UpdateProfileMarkPostHook(sync=self._sync,
                                      seprately_mark_for_iter=seprately_mark_for_iter))

        return ret

    def __getattr__(self, attr_name):
        return getattr(self.actual_optimizer, attr_name)

    def __setattr__(self, attr_name, value):
        setattr(self.actual_optimizer, attr_name, value)

class _MarkedProfileOptimizer(_MarkedProfileOptimizerBase):
    def __init__(self, actual_optimizer, sync=False, layerwise_sync=False):
        super(_MarkedProfileOptimizer, self).__init__(
            actual_optimizer, sync=sync, layerwise_sync=layerwise_sync)

    def setup(self, link):
        return self._setup(link, seprately_mark_for_iter=True)

class _MarkedProfileOptimizerForMN(_MarkedProfileOptimizerBase):
    def __init__(self, actual_optimizer, sync=False, layerwise_sync=False):
        super(_MarkedProfileOptimizerForMN, self).__init__(
            actual_optimizer, sync=sync, layerwise_sync=layerwise_sync)

    def setup(self, link):
        return self._setup(link, seprately_mark_for_iter=False)

    def update(self, lossfun=None, *args, **kwds):
        with prof.TimeRangeDecorator('iteration',
                                     sync=self._sync,
                                     argb_color=_itr_argb_color):
            ret = self.actual_optimizer.update(lossfun,
                                               *args,
                                               **kwds)
        return ret


def create_marked_profile_optimizer(
        actual_optimizer, sync=True, layerwise_sync=False):
    assert actual_optimizer is not None, 'actual_optimizer is required.'
    if issubclass(actual_optimizer.__class__, Optimizer):
        optimizer = _MarkedProfileOptimizer(actual_optimizer,
                                            sync=sync,
                                            layerwise_sync=layerwise_sync)
    else:
        optimizer = _MarkedProfileOptimizerForMN(actual_optimizer,
                                                 sync=sync,
                                                 layerwise_sync=layerwise_sync)

    return optimizer
