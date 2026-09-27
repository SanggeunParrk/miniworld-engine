"""Qualified H100 BF16 D32 training forward with FP32 base-2 LSE.

Unsupported inputs retain the existing implementation. No runtime compilation.
Set MINIWORLD_TRIATTN_TRAINING_FWD=0 to opt out for comparison or diagnosis.
"""
from __future__ import annotations
import functools
import hashlib
import importlib.util
import json
import os
from pathlib import Path
import sys

import torch

_ROOT=Path(__file__).resolve().parent
_EXT=None
SUPPORTED_LENGTHS=(384,768,1024)


@functools.lru_cache(maxsize=1)
def _artifact_ready():
    try:
        data=json.loads((_ROOT/'fwd_manifest.json').read_text())
        if data['python_abi']!=sys.implementation.cache_tag or data['torch']!=str(torch.__version__):
            return False
        for filename,digest in data['files'].items():
            if hashlib.sha256((_ROOT/filename).read_bytes()).hexdigest()!=digest:
                return False
        dispatcher=data['dispatcher']
        return hashlib.sha256((_ROOT/dispatcher['path']).read_bytes()).hexdigest()==dispatcher['sha256']
    except (OSError,ValueError,KeyError):
        return False


@functools.lru_cache(maxsize=None)
def _device_ready(device):
    return torch.cuda.get_device_capability(device)==(9,0) and 'H100' in torch.cuda.get_device_name(device)


def can_use(q,k,v,b):
    if os.environ.get('MINIWORLD_TRIATTN_TRAINING_FWD','1')=='0':
        return False
    if not(q.is_cuda and q.dtype==torch.bfloat16 and q.ndim==5 and q.shape[0]==1
           and q.shape[1]==4 and q.shape[4]==32 and q.shape[2] in SUPPORTED_LENGTHS
           and q.shape[3]==q.shape[2]):
        return False
    L=q.shape[2]
    return (all(t.device==q.device and t.dtype==q.dtype and t.shape==q.shape
                and t.stride(4)==1 and t.stride(1)==32 and t.stride(3)==128
                and t.stride(2)==L*128 and t.data_ptr()%16==0 for t in (q,k,v))
            and b.device==q.device and b.dtype==q.dtype and b.shape==(1,4,L,L)
            and b.is_contiguous() and b.data_ptr()%16==0 and _artifact_ready() and _device_ready(q.device))


def extension():
    global _EXT
    if _EXT is None:
        data=json.loads((_ROOT/'fwd_manifest.json').read_text())
        spec=importlib.util.spec_from_file_location(data['module_name'],_ROOT/data['binary'])
        _EXT=importlib.util.module_from_spec(spec);spec.loader.exec_module(_EXT)
    return _EXT


def forward(q,k,v,b):
    return tuple(extension().forward(q,k,v,b))
