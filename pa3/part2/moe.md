# Advantages of Mixture of Experts (MoE) Models

Based on the analysis of DeepSeek-V3's architecture and training costs, here are the key advantages of MoE models:

1. **Parameter Efficiency** : While Deepseek-V3 has a large number of total parameters due to its 256 routed experts, only a small fraction (8 experts per token) in active during inference. This allows MoE models to have massive model capacity while maintaining
reasonable computational costs. 

2. **Computational Cost Savings**: Only 8 out of 256 experts are used per token. The expert MLP size (2048) is much smaller than the standard MLP size (18432). This leads to lower Flops per token despite having too many experts. It reduces the training cost significantly compared to dense models of similar capacity. 

3. **Memory Efficiency**:  The peak memory usage during training is manageable because only active expert weights need to be loaded. This enables training larger models with the same hardware constraints

4. **Specialization and Flexibility** The large number of experts (256 routed + 1 shared) allows different experts to specialize in different types of inputs. The shared expert provides a backup for handling general cases

5. **Scaling Benefits**: MoE models can scale model capacity (through adding experts) without proportionally increasing compute costs/ This makes them particularly cost-effective for achieving better performance on a fixed training budget. 

Owing to these advantages might explian why Deepseek was able to train such a large and capable model within a $5M budget (just training, excluding the plannin phase cost). The MoE architecture provides an efficient way to scale model capacity while keeping computational and memory costs manageable.