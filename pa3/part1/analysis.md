Analysis of MoE Forward Pass Performance

(After running ! mpirun -n 4 python new_benchmark.py )

![](/part1/moe_forward_pass.png)

(Pls Check moe_forward_pass.png and new_benchmark.py)

From the benchmark results in the graph, we observe the following trends:

SimpleMoE exhibits a linear increase in forward pass time as batch size increases. This is expected since there is no parallelism, and computation scales directly with the input size.

MoE_TP (Tensor Parallelism) initially performs well for small batch sizes but scales worse than SimpleMoE at larger batch sizes. This suggests that communication overhead outweighs computation savings as batch size grows, leading to increased execution time.

MoE_EP (Expert Parallelism) consistently shows the lowest forward pass time across all batch sizes. Since each process handles a subset of experts, it distributes computation more efficiently, reducing the per-process workload and preventing bottlenecks.