"""Pin explicit Lt plans to the same wheel runtime under sanitizer injection."""
from pathlib import Path
import ctypes
import types


def pin_lt_library():
    import lt_contract
    path = Path('/workspace/envs/engine/lib/python3.10/site-packages/nvidia/cublas/lib/libcublasLt.so.12')
    library = ctypes.CDLL(str(path))
    if library.cublasLtGetVersion() != 120804:
        raise RuntimeError('Frozen Lt selections require cuBLASLt 12.8.4')
    # Only this experimental module's ctypes namespace is replaced. Do not alter
    # ctypes globally or remove any frozen algorithm identity assertions.
    namespace = types.SimpleNamespace(**vars(ctypes))
    def load(name, *args, **kwargs):
        return ctypes.CDLL(str(path) if name == 'libcublasLt.so.12' else name,
                           *args, **kwargs)
    namespace.CDLL = load
    lt_contract.C = namespace
    return str(path)
