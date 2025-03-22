import argparse
import json
import math
from scipy.optimize import minimize_scalar

def model_training_cost_analysis_llama(model_config_path):

    with open(model_config_path, 'r') as f:
        config = json.load(f)
    
    # Model parameters
    vocab_size = config['vocab_size']
    hidden_size = config['hidden_size']
    intermediate_size = config['intermediate_size']
    num_hidden_layers = config['num_hidden_layers']
    num_attention_heads = config['num_attention_heads']
    max_sequence_length = config['max_sequence_length']

    '''
    Model Parameters calculation
    '''

    # Embedding layer parameters - embedding parameters, no positional embeddings since RoPE is used
    embedding_params = vocab_size * hidden_size 

    # Transformer layer parameters: 2 x layernorm, attention, and mlp for each layer
    layernorm_params = 2 * 2 * hidden_size # one before attention and one befor mlp
    attention_params = 4 * hidden_size * hidden_size 
    mlp_params = 3 * intermediate_size * hidden_size 
    transformer_params = num_hidden_layers * (layernorm_params  + attention_params + mlp_params)
    
    # Final layer parameters: layer norm and linear layer
    final_layer_params = 2 * hidden_size + hidden_size * vocab_size

    # Total parameters
    total_params = embedding_params + transformer_params + final_layer_params

    '''
    TFLOPs calculation for a forward pass of a single transformer layer with b = 1
    '''
    # Calculate TFLOPs per layer with b = 1, taken from lecture slides
    attn_matmult_flops = 3 * 2 * max_sequence_length * hidden_size * hidden_size
    attn_softmax_flops = 2 * max_sequence_length * max_sequence_length * hidden_size \
                    + 3 * max_sequence_length * max_sequence_length * num_attention_heads
    attn_weighted_sum_flops = 2 * max_sequence_length * max_sequence_length * hidden_size
    output_matmult_flops = 2 * max_sequence_length * hidden_size * hidden_size

    mlp_flops = 6 * max_sequence_length * hidden_size * 4 * hidden_size # ( i = 4h)

    flops_layer_TF = (attn_matmult_flops + attn_softmax_flops + attn_weighted_sum_flops + \
                            output_matmult_flops + mlp_flops) / 1e12
    
    '''
    Peak memory calculation for forward pass of a single transformer layer with b = 1, fp16, and
    checkpoint rematerialization i.e only final output of MLP layer is stored
    '''

    # model weights
    layernorm_weights = 2 * 2 * hidden_size # (gamma, beta)
    attn_weights = 4 * hidden_size * hidden_size # (Q, K, V, O)
    mlp_weights = 2 * intermediate_size * hidden_size # (W1, W2)

    # final output of MLP layer, only stored as checkpoint rematerialization is used
    final_mlp_output = max_sequence_length * hidden_size

    peak_memory = (layernorm_weights + attn_weights + mlp_weights + final_mlp_output) * 2 # 2 for fp16
    peak_memory_GB = peak_memory / (1024 ** 3)

    return total_params, flops_layer_TF, peak_memory_GB

def model_training_cost_analysis_deepseek(model_config_path):
    with open(model_config_path, 'r') as f:
        config = json.load(f)
    
    '''
    Model Parameters calculation
    '''
    
    # Model parameters
    vocab_size = config['vocab_size']
    hidden_size = config['hidden_size']
    intermediate_size = config['intermediate_size']
    num_hidden_layers = config['num_hidden_layers']
    num_attention_heads = config['num_attention_heads']
    max_sequence_length = config['max_sequence_length']
    
    # MoE specific parameters
    n_routed_experts = config['n_routed_experts']
    n_shared_experts = config['n_shared_experts']
    moe_intermediate_size = config['moe_intermediate_size']
    num_experts_per_tok = config['num_experts_per_tok']

    # Embedding lyaer parameters
    embedding_params = vocab_size * hidden_size

    # Attention block parameters
    layernorm_params = 2 * 2 * hidden_size # 1 for pre-attention and 1 for post-attention
    attention_params = 4 * hidden_size * hidden_size
    
    # Shared MLP, Router, MoE parameters
    shared_mlp_params = n_shared_experts * (3 * intermediate_size * hidden_size)      # 3x for up & down projections
    router_params = hidden_size * n_routed_experts # gated routed parameters
    expert_mlp_params = n_routed_experts * (3 * moe_intermediate_size * hidden_size)  # 3x for up & down projections
    
    transformer_params = num_hidden_layers * (layernorm_params + attention_params + shared_mlp_params + 
                                                router_params + expert_mlp_params)
    
    # Final layer parameters
    final_layer_params = 2 * hidden_size + hidden_size * vocab_size

    # Total parameters
    total_params = embedding_params + transformer_params + final_layer_params

    '''
    TFLOPs calculation for a forward pass of a single transformer layer with b = 1
    '''

    # Attention block FLOPs
    attn_matmult_flops = 3 * 2 * max_sequence_length * hidden_size * hidden_size
    attn_softmax_flops = 2 * max_sequence_length * max_sequence_length * hidden_size + \
                        3 * max_sequence_length * max_sequence_length * num_attention_heads
    attn_weighted_sum_flops = 2 * max_sequence_length * max_sequence_length * hidden_size
    output_matmult_flops = 2 * max_sequence_length * hidden_size * hidden_size

    # MoE specific FLOPs
    router_flops = 2 * max_sequence_length * hidden_size * n_routed_experts
    expert_mlp_flops = num_experts_per_tok * 2 * max_sequence_length * hidden_size * moe_intermediate_size
    shared_mlp_flops = 2 * max_sequence_length * hidden_size * intermediate_size

    flops_layer_TF = (attn_matmult_flops + attn_softmax_flops + attn_weighted_sum_flops + 
                        output_matmult_flops + router_flops + expert_mlp_flops + shared_mlp_flops) / 1e12

    '''
    Peak memory calculation for forward pass of a single transformer layer with b = 1, fp16, and
    checkpoint rematerialization i.e only final output of MLP layer is stored
    '''

    # Peak memory calculation (fp16, with checkpoint rematerialization)
    layernorm_weights = 2 * 2 * hidden_size
    attn_weights = 4 * hidden_size * hidden_size
    expert_weights = n_routed_experts * 2 * moe_intermediate_size * hidden_size
    shared_weights = n_shared_experts * 2 * intermediate_size * hidden_size
    router_weights = hidden_size * n_routed_experts

    final_output = max_sequence_length * hidden_size

    peak_memory = (layernorm_weights + attn_weights + expert_weights + 
                  shared_weights + router_weights + final_output) * 2  # 2 for fp16
    peak_memory_GB = peak_memory / (1024 ** 3)

    return total_params, flops_layer_TF, peak_memory_GB

def get_optimal_N_D_from_cost(cost_budget):
    """
    cost_budget:  a monetary training budget (in dollars)
    Returns:
        N: Optimal total model parameters (in absolute numbers)
        D: Optimal number of training tokens (in absolute numbers)
        training_budget_flops: Effective total training FLOPs (in FLOPs)
        best_gpu: name of the selected GPU (one of 'A100', 'V100', 'T4')
    """
    # GPU options
    gpus = {
        'A100': {'cost_per_hour': 4.0, 'flops': 312e12},
        'V100': {'cost_per_hour': 2.5, 'flops': 125e12},
        'T4': {'cost_per_hour': 1.0, 'flops': 65e12}
    }

    mfu = 0.4  # Model utilization factor

    # Total traning seconds for each GPU
    total_seconds = {gpu: (cost_budget / gpus[gpu]['cost_per_hour']) * 3600 for gpu in gpus}

    # Total effective training FLOPs for each GPU
    total_flops = {gpu: total_seconds[gpu] * gpus[gpu]['flops'] * mfu for gpu in gpus}

    # Best GPU based on the maximum total FLOPs
    best_gpu = max(total_flops, key=total_flops.get)
    training_budget_flops = total_flops[best_gpu]

    # Given Scaling law constants
    a, b, c = 406.4, 410.7, 1.69

    # Loss function to minimize. Using total_budget_flops = 6ND constraint to change the scaling loss function 
    # in a single vaiable optimization problem. 6ND comes from the assumption 6 FLOPs is required to train per parameter per token, 
    # 2 for forward pass, and 4 for backpropagation.
    def loss(N):
        D = training_budget_flops/ (6 * N)
        return a / N**0.34 + b / D**0.29 + c
    
    # find optimal N using scipy's optimize
    result = minimize_scalar(loss, bounds=(1e5, 1e15))
    N = int(result.x)
    D = int(training_budget_flops / (6 * N))

    return N, D, training_budget_flops, best_gpu


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Model training cost analysis')
    parser.add_argument('--model_config', type=str, help='Path to model config file')
    parser.add_argument('--training_budget', type=float, default=None, help='Training budget')
    args = parser.parse_args()

    if args.model_config:
        if 'deepseek' in args.model_config:
            num_parameters, num_flops, memory_cost = model_training_cost_analysis_deepseek(args.model_config)
        elif 'llama' in args.model_config:
            num_parameters, num_flops, memory_cost = model_training_cost_analysis_llama(args.model_config)
        elif 'my_model' in args.model_config:
            num_parameters, num_flops, memory_cost = model_training_cost_analysis_llama(args.model_config)
        else:
            print('Unknown LLM Type!')
            exit()
        print(f"Number of parameters: {num_parameters}")
        print(f"Number of TFLOPs: {num_flops}")
        print(f"Peak memory cost: {memory_cost} GBs")

    if args.training_budget:    
        N, D, training_budget_flops, best_gpu = get_optimal_N_D_from_cost(args.training_budget)
        print(f"best_gpu: {best_gpu}")
        print(f"training_budget_flops: {training_budget_flops}")
        print(f"Optimal N: {N}")
        print(f"Optimal D: {D}")
