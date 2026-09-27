import torch, traceback
from miniworld_engine.kernels.drivers.mpnn_edge_tail import _graph
from miniworld_engine.kernels.mpnn_edge_tail.interface import edge_tail_update
t = _graph(64)
try:
    edge_tail_update(**t, eps=1e-5, dropout_probability=0.0, backend="triton_compute")
    print("forward ok")
except Exception:
    traceback.print_exc()
