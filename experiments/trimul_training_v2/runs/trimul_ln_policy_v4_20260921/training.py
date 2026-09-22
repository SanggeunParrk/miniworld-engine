"""Selected development TriMul training path: saved input x_n + output LN stats.

Raw tri stays as the sole large output-side B1 operand. Mean/rstd are saved by
K3 and prefetched by TMA with the B1 input tile. Output xhat is not saved.
Production promotion is separate: the existing B7 reference issue remains.
"""
from final import Replacement
class Training(Replacement):
 def __init__(self, inputs):
  super().__init__(inputs, -1)
