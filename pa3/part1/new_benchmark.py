import numpy as np
import matplotlib.pyplot as plt
from mpi4py import MPI
from rng import get_rng
from mpiwrapper import mpi
from moe import SimpleMoE, MoE_EP, MoE_TP
import time
import pandas as pd

def run_moe(moe_type="tp", batch_size=8, feature_dim=32, hidden_dim=128, output_dim=64, num_experts=None, topk=2):
    num_experts = mpi.get_size()
    np.random.seed(0)
    X = np.random.randn(batch_size, feature_dim)

    if moe_type != "simple":
        if mpi.get_rank() == 0:
            X = get_rng().randn(batch_size, feature_dim)
        else:
            X = None
        X = mpi.comm.bcast(X, root=0)

    model_class = {
        "simple": SimpleMoE,
        "ep": MoE_EP,
        "tp": MoE_TP
    }.get(moe_type, MoE_TP)

    moe = model_class(
        input_dim=feature_dim,
        hidden_dim=hidden_dim,
        output_dim=output_dim,
        num_experts=num_experts,
        topk=topk
    )

    _ = moe(X)  # Warm-up

    N = 3
    total_forward_time = 0
    total_comm_time = 0

    for _ in range(N):
        start_forward = time.time()
        outputs = moe(X)
        end_forward = time.time()

        start_comm = time.time()
        mpi.allreduce(outputs)  # Measure communication separately
        end_comm = time.time()

        total_forward_time += (end_forward - start_forward)
        total_comm_time += (end_comm - start_comm)

    avg_forward_ms = 1000 * total_forward_time / N
    avg_comm_ms = 1000 * total_comm_time / N

    if mpi.get_rank() == 0:
        print(f"{moe_type} MoE - Forward Pass: {avg_forward_ms:.2f} ms, Communication: {avg_comm_ms:.2f} ms")

    return avg_forward_ms, avg_comm_ms

def benchmark_moe():
    batch_sizes = [8, 16, 32, 64, 128]
    results = {
        "SimpleMoE": {"forward": [], "comm": []},
        "MoE_TP": {"forward": [], "comm": []},
        "MoE_EP": {"forward": [], "comm": []}
    }

    for batch_size in batch_sizes:
        fwd, comm = run_moe(moe_type="simple", batch_size=batch_size)
        results["SimpleMoE"]["forward"].append(fwd)
        results["SimpleMoE"]["comm"].append(comm)

        fwd, comm = run_moe(moe_type="tp", batch_size=batch_size)
        results["MoE_TP"]["forward"].append(fwd)
        results["MoE_TP"]["comm"].append(comm)

        fwd, comm = run_moe(moe_type="ep", batch_size=batch_size)
        results["MoE_EP"]["forward"].append(fwd)
        results["MoE_EP"]["comm"].append(comm)

    return batch_sizes, results

if __name__ == "__main__":
    batch_sizes, results = benchmark_moe()

    if mpi.get_rank() == 0:
        df_results = pd.DataFrame({
            "Batch Size": batch_sizes,
            "SimpleMoE_Forward": results["SimpleMoE"]["forward"],
            "SimpleMoE_Comm": results["SimpleMoE"]["comm"],
            "MoE_TP_Forward": results["MoE_TP"]["forward"],
            "MoE_TP_Comm": results["MoE_TP"]["comm"],
            "MoE_EP_Forward": results["MoE_EP"]["forward"],
            "MoE_EP_Comm": results["MoE_EP"]["comm"]
        })
        print(df_results)

        # Plot Forward Pass Time
        plt.figure(figsize=(8, 6))
        plt.plot(batch_sizes, results["SimpleMoE"]["forward"], marker='o', label="SimpleMoE Forward")
        plt.plot(batch_sizes, results["MoE_TP"]["forward"], marker='s', label="MoE_TP Forward")
        plt.plot(batch_sizes, results["MoE_EP"]["forward"], marker='^', label="MoE_EP Forward")
        plt.xlabel("Batch Size")
        plt.ylabel("Avg Forward Pass Time (ms)")
        plt.title("MoE Forward Pass Benchmark")
        plt.legend()
        plt.grid(True)
        plt.savefig("moe_forward_pass.png")
        plt.show()
