import pytest
import torch
from miniworld_engine.kernels.trimul_inproj.cuda._h100_pack import pack_into

pytestmark = [pytest.mark.gpu, pytest.mark.skipif(not torch.cuda.is_available(), reason='CUDA required')]

@pytest.mark.parametrize('d',[64,128,256,384,512])
@pytest.mark.parametrize('transposed',[False,True])
def test_pack_live_weights(d,transposed):
    weights=[torch.randn(2*d,d,device='cuda',dtype=torch.bfloat16) for _ in range(4)]
    if transposed:weights=[w.t().contiguous().t() for w in weights]
    out=torch.empty(8*d,d,device='cuda',dtype=torch.bfloat16)
    def ref():
        wl,wlg,wr,wrg=weights
        return torch.stack((torch.cat((wlg,wrg)).reshape(-1,32,d),torch.cat((wl,wr)).reshape(-1,32,d)),1).reshape(8*d,d)
    pack_into(out,*weights)
    torch.testing.assert_close(out,ref(),rtol=0,atol=0)
    stream=torch.cuda.Stream();stream.wait_stream(torch.cuda.current_stream())
    with torch.cuda.stream(stream):pack_into(out,*weights)
    torch.cuda.current_stream().wait_stream(stream)
    graph=torch.cuda.CUDAGraph()
    with torch.cuda.graph(graph,stream=stream):pack_into(out,*weights)
    for w in weights:w.add_(1)
    graph.replay()
    torch.testing.assert_close(out,ref(),rtol=0,atol=0)
