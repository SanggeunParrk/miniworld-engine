"""D512 short: overlap independent output-weight gradients with input backward.

DP/normalized-triangle and DG/normalized-input remain live until the side stream
joins. Their output matrices are disjoint from main-stream dNorm/dTriangle/dX.
PyTorch provides per-stream BLAS workspace for dWproj; frozen prefix-gate dW has
its own workspace. Main's Lt workspace is never shared with a side Lt call.
"""
import torch

class OutputWeightOverlap:
    def __init__(self, plan, phase='after_dn'):
        assert (plan.p.D,plan.p.n)==(512,384)
        assert plan.split_dwp is None
        assert plan.joint_choice is None
        assert 'dwp' not in plan.schedule.selected
        self.plan=plan;self.original=plan.backward;self.phase=phase
        self.side=torch.cuda.Stream(device=plan.p.x.device)
        self.ready=torch.cuda.Event();self.done=torch.cuda.Event()
        assert plan.prefix_gate_workspace.data_ptr()!=plan.schedule.workspace.data_ptr()
    def launch(self):
        p=self.plan.p;main=torch.cuda.current_stream();self.ready.record(main)
        with torch.cuda.stream(self.side):
            self.side.wait_event(self.ready)
            torch.mm(p.tensors[7].t(),p.tensors[6],out=p.dwp)
            self.plan.prefix_gate_dw()
            self.done.record(self.side)
    def __call__(self):
        plan=self.plan;p=plan.p;b=plan.b1;s=plan.schedule
        plan.delta_backward.launch();plan.delta_projection()
        b.epi.launch((1056,1,1),(256,1,1),[b.ep],0)
        if self.phase=='after_gate':self.launch()
        s.run('dn')
        if self.phase=='after_dn':self.launch()
        plan.ln()
        if self.phase=='after_ln':self.launch()
        for name in ('bc0','bc1','bc2','bc3'):s.run(name)
        plan.b7.source_only();plan.dx.copy_prefix();plan.dx.pack_weights();s.run('dx')
        torch.cuda.current_stream().wait_event(self.done)
        plan.dx.reduce_only()
        return p.outputs
