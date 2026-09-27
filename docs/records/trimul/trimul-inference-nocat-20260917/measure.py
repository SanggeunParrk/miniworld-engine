"""Measure only the replaced contraction block, separately from the complete module."""
import json
from pathlib import Path
import statistics
import torch
from miniworld_engine.kernels.trimul_inproj.triton.contract import packed_forward

@torch.no_grad()
def old(left,right,h):
    return torch.cat((torch.bmm(left[:h],right[:h].transpose(1,2)),torch.bmm(left[h:].transpose(1,2),right[h:])),0)

@torch.no_grad()
def capture(fn):
    stream=torch.cuda.Stream();stream.wait_stream(torch.cuda.current_stream())
    with torch.cuda.stream(stream):
        for _ in range(3):fn()
    torch.cuda.current_stream().wait_stream(stream)
    graph=torch.cuda.CUDAGraph()
    with torch.cuda.graph(graph):output=fn()
    return graph,output

def time_graph(graph):
    start,end=torch.cuda.Event(enable_timing=True),torch.cuda.Event(enable_timing=True)
    start.record()
    for _ in range(100):graph.replay()
    end.record();end.synchronize()
    return start.elapsed_time(end)/100

@torch.no_grad()
def main():
    rows=[]
    for l in (128,384,768):
        torch.manual_seed(917)
        x=torch.randn(256,l,l,device='cuda',dtype=torch.bfloat16);z=torch.randn_like(x)
        calls={'before':lambda:old(x,z,128),'after':lambda:packed_forward(x,z,128)}
        # Exclude cuBLAS first-use workspace allocation from both memory arms.
        for fn in calls.values():
            for _ in range(3):
                fn()
        torch.cuda.synchronize()
        peaks={}
        for name,fn in calls.items():
            torch.cuda.synchronize();torch.cuda.reset_peak_memory_stats();base=torch.cuda.memory_allocated()
            y=fn();torch.cuda.synchronize();peaks[name]=torch.cuda.max_memory_allocated()-base;del y
        graphs={name:capture(fn) for name,fn in calls.items()}
        torch.testing.assert_close(graphs['before'][1],graphs['after'][1],rtol=0,atol=0)
        samples={name:[] for name in calls}
        for r in range(10):
            for name in list(calls)[::(-1 if r%2 else 1)]:samples[name].append(time_graph(graphs[name][0]))
        ms={k:statistics.median(v) for k,v in samples.items()}
        row=dict(length=l,ms=ms,speedup=ms['before']/ms['after'],peak_extra_bytes=peaks,samples_ms=samples)
        rows.append(row);print(row,flush=True)
        del graphs,x,z
    (Path(__file__).parent/'contraction-benchmark.json').write_text(json.dumps({'scope':'contraction block only; BF16 H100 h128 B1, CUDA graph, 10 alternating rounds x100 replays; not full-module timing','rows':rows},indent=2))
if __name__=='__main__':main()
