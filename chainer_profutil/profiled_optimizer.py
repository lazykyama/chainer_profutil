

from chainer.optimizer import Optimizer
from chainer.function_hooks import CUDAProfileHook

from cupy import cuda
from cupy import prof
from cupy.cuda import runtime


_fwd_argb_color = 0xff76b900
_bwd_argb_color = 0xff7fdbff
_upd_argb_color = 0xffe60012

def _try_to_sync_if_needed(self):
    if self._sync:
        runtime.deviceSynchronize()

def _colored_range_push_if_exists(msg, argb_color):
    if argb_color is None:
        cuda.nvtx.RangePush(msg)
    else:
        cuda.nvtx.RangePushC(msg, argb_color)

class UpdateProfileMarkPreHook(object):
    name = 'preupdate'
    call_for_each_param = False
    timing = 'pre'

    def __init__(self, sync=True, argb_color=None):
        self._sync = sync
        self._argb_color = argb_color

    def __call__(self, rule):
        _try_to_sync_if_needed(self)
        _colored_range_push_if_exists('update', self._argb_color)

class UpdateProfileMarkPostHook(object):
    name = 'postupdate'
    call_for_each_param = False
    timing = 'post'

    def __init__(self, sync=True):
        self._sync = sync

    def __call__(self, rule):
        _try_to_sync_if_needed(self)
        cuda.nvtx.RangePop()


class FwdBwdProfileMarkHook(CUDAProfileHook):

    name = 'FwdBwdProfileMarkHook'

    def __init__(self, sync=True, argb_color=None):
        super(FwdBwdProfileMarkHook, self).__init__()
        self._sync = sync
        self._argb_color = argb_color

    def forward_preprocess(self, function, in_data):
        _try_to_sync_if_needed(self)
        _colored_range_push_if_exists(
            function.label + '.forward', self._argb_color)

    def forward_postprocess(self, function, in_data):
        _try_to_sync_if_needed(self)
        cuda.nvtx.RangePop()

    def backward_preprocess(self, function, in_data, out_grad):
        _try_to_sync_if_needed(self)
        _colored_range_push_if_exists(
            function.label + '.backward', self._argb_color)

    def backward_postprocess(self, function, in_data, out_grad):
        _try_to_sync_if_needed(self)
        cuda.nvtx.RangePop()


def _add_backward_mark(func, sync, layerwise_sync):
    def backward_wrapper(*args, **kwargs):
        with prof.TimeRangeDecorator('model.backward', sync=sync, argb_color=_bwd_argb_color):
            with FwdBwdProfileMarkHook(sync=layerwise_sync, argb_color=_bwd_argb_color):
                ret = func(*args, **kwargs)
        return ret
    return backward_wrapper

def make_wrapped_lossfunc(func, sync=True, layerwise_sync=False):
    if func is None:
        raise ValueError('func is required.')

    def forward_wrapper(*args, **kwargs):
        with prof.TimeRangeDecorator('model.forward', sync=sync, argb_color=_fwd_argb_color):
            with FwdBwdProfileMarkHook(sync=layerwise_sync, argb_color=_fwd_argb_color):
                ret = func._org_forward(*args, **kwargs)
        ret.backward = _add_backward_mark(ret.backward, sync, layerwise_sync)
        return ret

    func._org_forward = func.forward
    func.forward = forward_wrapper
    return func

def _setup(self, link):
    make_wrapped_lossfunc(
        link,
        sync=self.sync_for_prof,
        layerwise_sync=self.layerwise_sync_for_prof)

    ret = super(self.__class__, self).setup(link)
    self.add_hook(UpdateProfileMarkPreHook(
        sync=self.sync_for_prof, argb_color=_upd_argb_color))
    self.add_hook(UpdateProfileMarkPostHook(sync=self.sync_for_prof))
    return ret


def create_marked_profile_optimizer(
        basecls, sync=True, layerwise_sync=False):
    assert basecls, 'basecls is required.'
    if not issubclass(basecls, (Optimizer, )):
        raise RuntimeError('{} may not be Chainer\'s optimizer.')

    MarkedProfileOptimizer = type(
        'MarkedProfileOptimizer',
        (basecls, ),
        {'setup': _setup})
    def make_instance(*args, **kwargs):
        optimizer = MarkedProfileOptimizer(*args, **kwargs)
        optimizer.sync_for_prof = sync
        optimizer.layerwise_sync_for_prof = layerwise_sync
        return optimizer

    return make_instance
