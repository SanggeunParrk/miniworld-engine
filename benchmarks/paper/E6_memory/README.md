# E6: peak memory

The E2 grid (both classes, inference and training) x every backend (`pytorch`, `cuda`, `triton`, `cueq`; `anthropic` on the kit stack).
`paper_common/memory.py`: module and inputs built as the speed measurer does, the step (`speed.make_step`: eval forward under no_grad / train forward +
backward) run once warm, cache emptied, `torch.cuda.reset_peak_memory_stats()`, one step, `torch.cuda.max_memory_allocated()`. Eager step on the default stream
(no CUDA graph, whose private pool would hide the allocator's peak). `peak_mem_gib` = that peak in absolute terms (it includes inputs and parameters);
the input tensors, the parameters and the memory in use before the step are separate columns, `step_extra_gib = peak - base`. OOM = `skipped`.
Single deterministic measurement per case (repeat files are aggregated by taking the last).

    cd benchmarks/paper && paper_common/chain.sh 6 E6_memory/h100.sbatch ; python E6_memory/collect.py --card h100

Output `E6_<card>.csv`: common columns (`peak_mem_gib` filled, ms empty) + `stack,row,variant,samples,depth,base_gib,input_gib,param_gib,step_extra_gib`.
