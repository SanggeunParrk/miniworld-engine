import json
from miniworld_engine.kernels.transition.cuda.variants import extension
c=dict(bk=384,bn=32,bo=64,mgroups=1,ngroups=2,stages=1,min_blocks=1)
e=extension('full_k',384,c);print(json.dumps(dict(config=c,extension=e.__file__,resources=e.resources())),flush=True)
