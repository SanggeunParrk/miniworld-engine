"""Development adapter: save FP32 pre-affine output xhat + rstd.
B1 receives no raw tri and does not recompute output LN normalization.
The inherited L768 B7 strict-reference issue still blocks production promotion.
"""
from bench import Replacement
class Training(Replacement):
 def __init__(self,a):super().__init__(a,1)
