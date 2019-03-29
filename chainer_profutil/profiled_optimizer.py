

import enum

from chainer.optimizer import Optimizer
from chainer.function_hooks import CUDAProfileHook

from cupy import cuda
from cupy import prof
from cupy.cuda import runtime


_itr_argb_color = 0xfffff100
_fwd_argb_color = 0xff76b900
_bwd_argb_color = 0xff7fdbff
_upd_argb_color = 0xffe60012


class SyncLevel(enum.IntEnum):
    COARSEST = 1
    SECOND = 2
    FINEST = 3


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


class UpdateProfileMarkHookMixin(object):
    def _need_update_sync(self):
        if not self._sync:
            return False
        else:
            return (self._sync_level >= SyncLevel.SECOND)

class UpdateProfileMarkPreHook(UpdateProfileMarkHookMixin):
    name = 'preupdate'
    call_for_each_param = False
    timing = 'pre'

    def __init__(self, sync, sync_level, argb_color):
        self._sync = sync
        self._sync_level = sync_level
        self._argb_color = argb_color

    def __call__(self, rule):
        upd_sync = self._need_update_sync()
        range_push(upd_sync, 'model.update', self._argb_color)

class UpdateProfileMarkPostHook(UpdateProfileMarkHookMixin):
    name = 'postupdate'
    call_for_each_param = False
    timing = 'post'

    def __init__(self, sync, sync_level, seprately_mark_for_iter):
        self._sync = sync
        self._sync_level = sync_level
        self._seprately_mark_for_iter = seprately_mark_for_iter

    def __call__(self, rule):
        upd_sync = self._need_update_sync()
        if not self._sync:
            itr_sync = False
        else:
            itr_sync = (self._sync_level >= SyncLevel.COARSEST)

        range_pop(upd_sync)  # pop 'model.update'
        if self._seprately_mark_for_iter:
            range_pop(itr_sync)  # pop 'iteration'


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


class VariableWrapper(object):
    def __init__(self, loss, sync, sync_level):
        self._loss = loss
        self._sync = sync
        self._sync_level = sync_level

    def backward(self, *args, **kwargs):
        if not self._sync:
            bwd_sync = False
            bwd_each_sync = False
        else:
            bwd_sync = (self._sync_level >= SyncLevel.SECOND)
            bwd_each_sync = (self._sync_level >= SyncLevel.FINEST)

        with prof.time_range('model.backward', sync=bwd_sync, argb_color=_bwd_argb_color):
            with FwdBwdProfileMarkHook(sync=bwd_each_sync, argb_color=_bwd_argb_color):
                ret = self._loss.backward(*args, **kwargs)
        return ret


def make_wrapped_link(link,
                      sync=True,
                      sync_level=SyncLevel.COARSEST,
                      seprately_mark_for_iter=True):
    assert SyncLevel.COARSEST <= sync_level <= SyncLevel.FINEST, \
        'Unexpected sync_level: {}'.format(sync_level)
    if link is None:
        raise ValueError('link is required.')
    if not hasattr(link, 'forward'):
        raise RuntimeError('link must have forward method.')

    def forward_wrapper(*args, **kwargs):
        if seprately_mark_for_iter and sync_level >= SyncLevel.COARSEST:
            range_push(sync, 'iteration', _itr_argb_color)

        if not sync:
            fwd_sync = False
            fwd_each_sync = False
        else:
            fwd_sync = (sync_level >= SyncLevel.SECOND)
            fwd_each_sync = (sync_level >= SyncLevel.FINEST)

        with prof.time_range('model.forward', sync=fwd_sync, argb_color=_fwd_argb_color):
            with FwdBwdProfileMarkHook(sync=fwd_each_sync, argb_color=_fwd_argb_color):
                loss = link._org_forward(*args, **kwargs)
        return VariableWrapper(loss, sync, sync_level)

    link._org_forward = link.forward
    link.forward = forward_wrapper
    return link


class _MarkedProfileOptimizerBase(object):
    def __init__(self, actual_optimizer, sync, sync_level):
        super(_MarkedProfileOptimizerBase, self).__setattr__(
            'actual_optimizer', actual_optimizer)
        super(_MarkedProfileOptimizerBase, self).__setattr__(
            '_sync', sync)
        super(_MarkedProfileOptimizerBase, self).__setattr__(
            '_sync_level', sync_level)

    def _setup(self, link, seprately_mark_for_iter=True):
        make_wrapped_link(
            link,
            sync=self._sync,
            sync_level=self._sync_level,
            seprately_mark_for_iter=seprately_mark_for_iter)
        ret = self.actual_optimizer.setup(link)

        self.actual_optimizer.add_hook(
            UpdateProfileMarkPreHook(sync=self._sync,
                                     sync_level=self._sync_level,
                                     argb_color=_upd_argb_color))
        self.actual_optimizer.add_hook(
            UpdateProfileMarkPostHook(sync=self._sync,
                                      sync_level=self._sync_level,
                                      seprately_mark_for_iter=seprately_mark_for_iter))

        return ret

    def __getattr__(self, attr_name):
        return getattr(self.actual_optimizer, attr_name)

    def __setattr__(self, attr_name, value):
        setattr(self.actual_optimizer, attr_name, value)

class _MarkedProfileOptimizer(_MarkedProfileOptimizerBase):
    def __init__(self, actual_optimizer, sync, sync_level):
        super(_MarkedProfileOptimizer, self).__init__(
            actual_optimizer, sync, sync_level)

    def setup(self, link):
        return self._setup(link, seprately_mark_for_iter=True)

class _MarkedProfileOptimizerForMN(_MarkedProfileOptimizerBase):
    def __init__(self, actual_optimizer, sync, sync_level):
        super(_MarkedProfileOptimizerForMN, self).__init__(
            actual_optimizer, sync, sync_level)

    def setup(self, link):
        return self._setup(link, seprately_mark_for_iter=False)

    def update(self, lossfun=None, *args, **kwds):
        if self._sync:
            iter_sync = True
        with prof.time_range('iteration',
                             sync=iter_sync,
                             argb_color=_itr_argb_color):
            ret = self.actual_optimizer.update(lossfun,
                                               *args,
                                               **kwds)
        return ret


def create_marked_profile_optimizer(
        actual_optimizer,
        sync=True,
        sync_level=SyncLevel.COARSEST):
    assert actual_optimizer is not None, 'actual_optimizer is required.'
    assert SyncLevel.COARSEST <= sync_level <= SyncLevel.FINEST, \
        'Unexpected sync_level: {}'.format(sync_level)

    if issubclass(actual_optimizer.__class__, Optimizer):
        optimizer = _MarkedProfileOptimizer(actual_optimizer,
                                            sync=sync,
                                            sync_level=sync_level)
    else:
        optimizer = _MarkedProfileOptimizerForMN(actual_optimizer,
                                                 sync=sync,
                                                 sync_level=sync_level)

    return optimizer
