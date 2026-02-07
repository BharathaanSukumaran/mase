# ADLS Labs – Group 15

---

## Introduction

---

## Lab 0 – Introduction and Environment Setup

### Setup

### Results

### Observations

### Conclusion

---

## Lab 1 – Quantisation and Pruning Fundamentals

### Overview
We investigate how we can make our models more efficient by quantising and pruning. These are common compression techniques used to reduce memory and compute costs. <br>
- Quantisation reduces the numerical precision used to represent weights and activations. 
- Sparsity refers to the number of model parameters that are set to zero and so do not contribute to computation.

### Quantisation
We focus on fixed-point quantisation in this lab. This is where floating-point weights and activations are represented using a limited number of bits. We compare Post-Training Quantisation (PTQ) and Quantisation-Aware Training (QAT). PTQ applies quantisation after training without modifying the model. QAT is when quantisation effects are simulated during training to allow the model to adapt. We sweep bit-widths from 4 to 32 showing how reduced precision degrades accuracy and how QAT can recover performance at lower bit-widths compared to PTQ. <br>

QAT introduces quantisation effects during training, then we retrain the model for one epoch. This allows the models parameters to be adjusted to the noise quantisation may create. 
Quantisation width is the number of bits used to represent a number. We vary the width and evaluate how this affects the models accuracy. An increased quantisation width will clearly increase the accuracy as there are fewer rounding errors and therefore higher precision. Here we compare the bit lengths `[4, 8, 16, 32]` for both PTQ and QAT.

![](images/Lab1-Quantisation-Results.png)
*How quantisation width affects model accuracy for Post-Training Quantisation (PTQ) and Quantisation-Aware Training (QAT)*

We see the largest difference between PTQ and QAT at lower quantisation widths. QAT has a much higher accuracy at 0.79256 with 4 bit-width compared to PTQ's 0.67572 accuracy. In general QAT performs better than PTQ at every bit width showing that it is a better and more resilient quantisation method. 

The fractional width is derived from the total bit-width while keeping the integer width approximately fixed in order to preserve dynamic range and ensure that accuracy changes primarily reflect quantisation precision rather than overflow effects.

```python
def make_quantization_config(w: int):
    frac_width = max(4, w-2)
    return {
        "by": "type",
        "default": {
            "config": {
                "name": None,
            }
        },
        "linear": {
            "config": {
                "name": "integer",
                # data
                "data_in_width": w,
                "data_in_frac_width": frac_width,
                # weight
                "weight_width": w,
                "weight_frac_width": frac_width,
                # bias
                "bias_width": w,
                "bias_frac_width": frac_width,
            }
        },
    }
```
### Pruning
Pruning is a method for increasing model sparsity by removing less important parameters. Higher sparsity corresponds to a larger fraction of weights being removed and we vary the sparsity level in this part of the lab. We can evaluate using random pruning, which removes weights at random or L1-norm pruning, which removes weights with the smallest magnitudes. We find that rule based pruning preserves accuracy better than random pruning, especially as we try and push our sparsity to a high level. <br>

We enforce the same sparsity across both weights and activations, with pruning performed locally so that each tensor is pruned independently based on its own metadata. Sparsity is swept from 0.1 to 0.9 using fine granularity to identify the point at which accuracy degrades sharply

Pruning has a much higher computational demand compared to quantisation. To combat any error and repeated execution caused by this, we added checkpoints after each run, a cooldown period between each run and explicit memory clean up.

All pruning experiments include one epoch of post-pruning fine-tuning as retraining is generally required to recover accuracy after parameters have been removed, much like in QAT.

![](images/Lab1-Pruning-Results.png)
*How Random and L1-Pruning affect model evaluation accuracy at a sweeping range of sparsity levels.*

Pruning is applied incrementally at each sparsity level, and the pruned model is evaluated on the IMDb dataset to measure the highest model accuracy. The results show that accuracy decreases as sparsity increases for both methods, but L1-norm pruning consistently preserves accuracy better than random pruning, particularly at higher sparsity levels. This gap widens as sparsity approaches extreme values, demonstrating that structured pruning strategies are more robust when aggressively compressing the model. Here pruning effects seem to be more gradual than quantisation effects observed in Lab 1.

---

## Lab 2 – Neural Architecture Search (NAS) and Compression-Aware Optimisation

### Overview

This lab constructs the NAS workfow for BERT, then shows how to train, evaluate and then finally compress the (best) discovered architecture. <br>
1. We start by initializing training pipeline (tokenization into input IDs and attention masks).
2. Next define a dictionary of search hyperparameters that BERT is allowed to change, such as number of transfomer layers, or even the type of layer.
3. A `construct_model` function is defined which samples the hyperparametrs from Optuna and constructs a BERT model using those settings. The evaluation of the sampling strategy from Optuna is the implementation focus of this lab.
4. An `objective` function is defined which essentially just evaluates the validation accuracy.
5. Finally, run multiple trials of each sampler and measure the accuracy across trials for each one.
6. Using the model from the sampler that yielded the best results, we finally compress the model using MASE. We may also fine-tune this model. We check how much the performance changed after the compression.

### Task 1

### Task 2
---

## Lab 3 – Mixed-Precision Quantisation Search

In this lab, we use Optuna which allows us to perform neural architecture search, building upon tutorial 6 which shows us how to quantize the `LinearInteger` using a low-precision config
```python
kwargs["config"] = {
    "data_in_width" : 8,
    "data_in_frac_width": 4,
    "weight_width": 8,
    "weight_frac_width": 4,
    "bias_width": 8,
    "bias_frac_width": 4,
}
```

This config specifies a *Q4.4* fixed-point format. 4 bits for the whole number, 4 bits for the fractional part. 
- `data_in`: quantize activations coming in from previous layer
- `weight_width`: quantize weight vector
- `bias_width`: quantize bias vector

1. Because different layers in the model have different senstivity to quantization schemes, it is more optimal to search over different bit widths when finding the best model. 

2. Additionally, we only considered the `LinearLayer` in tutorial 6. We also extend the search to consider all supported precision formats in MASE.

For both these extensions, we plot a graph showing how increasing the number of trials changes with the highest value of the obhjective function at the end of each set of trials.

### Setup

We add code to allow easy setup of configs for ranges and different layer choices in the following way

1. Set global dicts for the variables we will later pass Optuna's search space. We organise the layers by precision to allow setting up different studies for each precision.
```py
SEARCH_RANGES = {
    "width": [8, 16, 32],
    "frac_width": [2, 4, 8],
    "exponent_width": [2, 3, 4],
    "exponent_bias": [3, 7, 15],
    "block_size": [16, 32],
}

PRECISION_MAP = {
    "Integer": [LinearInteger],
    "Minifloat": [LinearMinifloatDenorm, LinearMinifloatIEEE],
    "Block": [LinearBlockFP, LinearBlockMinifloat, LinearBlockLog],
    "Binary": [LinearBinary],
    "Log": [LinearLog],
    "Full": [torch.nn.Linear],
}
```

2. We then have a function which receives a given precision layer, and configures its parameters, using the `SEARCH_RANGES` dict. 
```py
def configure_layer_params(
    trial: optuna.Trial, 
    layer_name: str, 
    layer_cls: Type, 
    base_kwargs: Dict[str, Any]
) -> Dict[str, Any]:
    """
    Populates the 'config' dictionary for a quantized layer based on Optuna suggestions.
    """
```

Different layers need different parameters, and that is handled within this function. Here is an example for `LinearInteger` and `LinearMinifloatDenorm`, `LinearMinifloatIEEE` layers. 

```python
    if layer_cls == LinearInteger:
        for comp in components:
            w = trial.suggest_categorical(f"{layer_name}_{comp}_width", SEARCH_RANGES["width"])
            fw = trial.suggest_categorical(f"{layer_name}_{comp}_frac_width", SEARCH_RANGES["frac_width"])
            config[f"{comp}_width"] = w
            config[f"{comp}_frac_width"] = fw

    elif layer_cls in [LinearMinifloatDenorm, LinearMinifloatIEEE]:
        for comp in components:
            w = trial.suggest_categorical(f"{layer_name}_{comp}_width", SEARCH_RANGES["width"])
            ew = trial.suggest_categorical(f"{layer_name}_{comp}_exp_width", SEARCH_RANGES["exponent_width"])
            config[f"{comp}_width"] = w
            config[f"{comp}_exponent_width"] = ew
            
            if layer_cls == LinearMinifloatDenorm:
                eb = trial.suggest_categorical(f"{layer_name}_{comp}_exp_bias", SEARCH_RANGES["exponent_bias"])
                config[f"{comp}_exponent_bias"] = eb
            else:
                # IEEE bias: 2^(k-1) - 1
                config[f"{comp}_exponent_bias"] = (2 ** (ew - 1)) - 1
```

3. We then define a `construct_model` function which with the `trial` also receives the specific `layer_choices` list for Optuna to use, replaces the `nn.torch.Linear` layer with the new layer after copying over weight and bias data from the old layer.
```python
def construct_model(trial: optuna.Trial, layer_choices) -> Any:
    trial_model = deepcopy(BASE_MODEL)

    for name, layer in trial_model.named_modules():
        if isinstance(layer, torch.nn.Linear):
            choice_idx = trial.suggest_int(f"{name}_type_idx", 0, len(layer_choices) - 1)
            new_layer_cls = layer_choices[choice_idx]

            if new_layer_cls == torch.nn.Linear:
                continue

            kwargs = {
                "in_features": layer.in_features,
                "out_features": layer.out_features,
                "bias": layer.bias is not None
            }

            kwargs = configure_layer_params(trial, name, new_layer_cls, kwargs)

            try:
                new_layer = new_layer_cls(**kwargs)
                new_layer.weight.data = layer.weight.data
                if layer.bias is not None and new_layer.bias is not None:
                    new_layer.bias.data = layer.bias.data
                
                deepsetattr(trial_model, name, new_layer)
                
            except Exception as e:
                print(f"Failed to construct {name} with {new_layer_cls}: {e}")
                raise e

    return trial_model
```

4. We then need an objective function which receives the `layer_choices` and constructs the objective function from that.
```py
def get_objective(layer_choices):
    def objective(trial):
        model = construct_model(trial, layer_choices)

        trainer = get_trainer(
            model=model,
            tokenized_dataset=dataset,
            tokenizer=tokenizer,
            evaluate_metric="accuracy",
            num_train_epochs=1,
        )
        
        if (LinearBlockMinifloat not in layer_choices) and (LinearBinaryResidualSign not in layer_choices):
            trainer.train()
        
        
        metrics = trainer.evaluate()
        accuracy = metrics.get("eval_accuracy")
        
        return accuracy
    
    return objective
```

`LinearBlockMinifloat` and `LinearBinaryResidualSign` do not define a backward pass so we cannot retrain the model if any of those layers are present.

5. Loop through a set of trials and record the best value from each study.
```py
N_TRIALS = 24

def run_comparison():
    plt.figure(figsize=(10, 6))
    
    # Iterate over each precision type (Integer, Minifloat, etc.)
    for name, layer_choices in PRECISION_MAP.items():
        print(f"--- Running Search for {name} Precision ---")
        
        sampler = RandomSampler()
        study = optuna.create_study(direction="maximize", sampler=sampler)
        
        best_values = []

        for n in range(1, N_TRIALS+1):
            study.optimize(get_objective(layer_choices), n_trials=n)
            best_values.append(study.best_value)
            
        plt.plot(range(1, len(best_values) + 1), best_values, label=name, marker='o')

    plt.xlabel("Number of Trials")
    plt.ylabel("Best Accuracy Achieved")
    plt.title("Mixed Precision Search Comparison")
    plt.legend()
    plt.grid(True)
    plt.show()
```

6. Apply MASE's compression pass to apply the best hyperparameters to integer quantization from task 1. We also apply pruning, a method for reducing model size and complexity by removing random weights and structural components.  



### Results

![](images/tutorial6_task1_integer_layerwise_running_best.png)
*Increasing number of trials per study for parameter search over `LinearInteger` layer*

![](images/tutorial6_task2_all_precisions_running_best.png)
*Increasing number of trials per study for parameter search over multiple precision layers*

For pruning, we apply integer quantization using the best hyperparameters from Optuna's search, and use a pruning configuration as follows
- *sparsity*: 0.5
- *method*: l1-norm
- *scope*: local

We analyse 2 variants of the compressed model, one that is post-trained and one that is not. We observed an accuracy of 0.502 without post-training, and one of 0.862096 with post-training.

### Observations

In task 1, we see an increase in the study's best value as we increase the number of trials per study. Another thing to note is that there is no added benefit in more trials between 8 - 18 trials, and 4-7 trials. 

In task 2, we compare different precisions as shown. We see that the `LinearBlockMinifloat` and `LinearBinaryResidualSign` layers started off with the lowest with 1 trial. This is explainable by the fact that with these layers, the model does not get retrained, and therefore does not adapt well to the change of precision. With more trials the `LinearBinaryResidualSign` layer does not improve, whereas the `LinearBlockMinifloat` does, going from a best accuracy of 0.50 to one of 0.84 at 12 trials. This is still underperforming in comparison to the models that were retrained.

The improvement in `LinearBlockMinifloat` is explainable by the fact that it has tunable hyperparameters that control information loss
```py
blk = trial.suggest_categorical(f"{layer_name}_block_size", SEARCH_RANGES["block_size"])

for comp in components:
    if layer_cls == LinearBlockMinifloat:
        config[f"{comp}_width"] = trial.suggest_categorical(f"{layer_name}_{comp}_width", SEARCH_RANGES["width"])
        config[f"{comp}_block_size"] = [blk] # block size made a list here to circumvent errors
        
        config[f"{comp}_exponent_width"] = 4
        config[f"{comp}_exponent_bias_width"] = 4
```

Here, we control the `data_in`, `weight`, and `bias` width and block size. With lower trials, Optuna might pick a configuration which destroys the model accuracy, but with higher trials, it has a chance to optimise for better configurations.

`LinearBinaryResidualSign` does not have such parameters, so Optuna has nothing to feedback loop to optimise over.

```py
    elif layer_cls == LinearBinaryResidualSign:
        config["data_in_stochastic"] = False
        config["weight_stochastic"] = False

        config["data_in_bipolar"] = True
        config["weight_bipolar"] = True

        config["binary_training"] = True
```


The next best performers are the `LinearBinaryScaling` and `LinearBinary` layers which have similar performance over the studies. 

With these precisions, we perform Quantization-aware training, allowing the model to adapt to the weights being only within the [-1,1] range.

The best performers are the `LinearBlockLog`, `LinearBlockFP`, `LinearLog`, `LinearMinifloatDenorm`, `LinearMinifloatIEEE` and `LinearInteger` layers.

### Conclusion

The fact that `LinearBinaryScaling` and `LinearBinary` outperform `LinearBLockMinifloat` could indicate that Optuna chose a configuration for `LinearBLockMinifloat` that is problematic. For instance, if the exponent width is too large, then the there's less mantissa bits, hence less precision, so very small weights in the block vanish to 0, whereas if the exponent width is too small, then very large weights in the block get clipped.

We also see the importance of hyperparameter tuning in choosing the optimal model as seen with results for `LinearBlockMinifloat` vs `LinearBinaryResidualSign`.

The best performers are so, even with `n_trials = 1`, because my search includes width = `[8, 16, 32]`. Most models lose little accuracy when quantized to 8-bit formats, and we give Optuna the opportunity to even go high precision than that. These precisions are still high enough to match the distribution of weights in the entire model. 

Moreover, these precisions preserve the algebraic properties of the original model, and preserve sign and magnitude. In comparison with the `LinearBinary` layer, a weight of 0.0001 and 5.0 both become 1.0 post quantization. The magnitude information has been lost.

Finally, we see the usefulness of post-training as shown by the compression results. Quantization and pruning is lossy, pre-training allows the model to adapt the model's weights to the new quantization scheme and structure imposed after quantization and pruning.

### Key Takeaways

Quantization is a process that allows model sizes to reduce, allowing for more efficient inference. This process is lossy, but through principled techniques like QAT and post-training, we can regain the lost information, sometimes even  surpursing the accuracy of the base model. Additinally, the hyperparameters chosen for any layer affect the performace, because they relate to how well the layer retains information. One can also pick precisions such that we effectively get a compression that doesn't loose much of the information compared to the full-precision model.

---

## Lab 4 – System Performance Tuning & Enhancement

### Overview

This lab investigates system-level performance optimisation for deep learning workloads in PyTorch, progressing from high-level compiler-based techniques to low-level custom kernel implementations. The experiments are structured to highlight the trade-offs between automation, algorithmic kernel design, and hardware-aware optimisation.

Specifically, we study:
1. Automatic graph-level optimisation using `torch.compile`.
2. Manual kernel fusion through the fused Scaled Dot-Product Attention (SDPA) operator.
3. Custom CUDA kernels for MXINT8 dequantisation and their impact on latency and GPU memory usage.

All experiments are evaluated from a systems perspective, with attention to benchmarking methodology, hardware constraints, and practical deployment implications.

---

### Setup

**Hardware**
- GPU: NVIDIA GeForce RTX 2050 (4 GB)
- CPU: 13th Gen Intel(R) Core(TM) i5-13420H
- Memory: 16GB

**Software**
- OS: Ubuntu 24.04.3 LTS
- PyTorch: 2.10.0+cu128
- CUDA: 12.8
- Transformers: 5.1.0

**Benchmarking protocol**
- Inference-only measurements using `model.eval()` and `torch.no_grad()`.
- Warm-up iterations executed before timing.
- GPU timing performed with CUDA events and explicit synchronization.
- Data loading, tokenization, and host–device transfers excluded from timed regions.

---

### Part A – Automatic Optimisation with `torch.compile`

#### Background

`torch.compile` is a just-in-time (JIT) compilation framework introduced in PyTorch 2.0. It captures Python-level model execution, constructs a graph representation, and lowers this graph to optimised kernels. The compilation pipeline consists of three main components:

- **TorchDynamo**, which captures PyTorch programs via Python bytecode interception.
- **TorchInductor**, which performs graph rewriting, operator fusion, and kernel generation using Triton, a specialized language and compiler for GPU programming (to define custom GPU kernels).
- **AOT Autograd**, which enables whole-graph capture across forward and backward passes.

The PyTorch documentation notes that the first executions of a compiled model are expected to be slower due to compilation overhead, and that larger speedups are more readily observed on datacenter-class GPUs.


#### Question: Why can `torch.compile` appear slower, and when does it help?

In early experiments, the compiled model sometimes appeared slower than the eager (uncompiled) model. This behaviour is expected and arises from several factors:

1. **Compilation overhead**  
   Initial executions include graph capture and kernel generation. For small batch sizes or short benchmark runs, this overhead dominates total runtime.

2. **Benchmarking artifacts**  
   Timing utilities, CUDA synchronization, and Python-level overhead can distort measurements if included inside the timed region. Mixing warm-up and steady-state iterations further biases results against the compiled path.

3. **Hardware limitations**  
   The RTX 2050 is a consumer-grade GPU with limited memory bandwidth and shared-memory capacity. As a result, the achievable speedups are smaller than those reported on datacenter GPUs such as V100, A100, or H100.

When benchmarking is performed correctly and the model is executed in steady state, `torch.compile` provides consistent but modest speedups on both CPU and GPU.

#### Benchmarking methodology and design decisions

The starter benchmarking code measured individual forward passes inside a Python loop and experimented with compiling timing utilities themselves. We found this approach introduced avoidable overhead and high variance, particularly on GPU, where per-iteration synchronization dominated the measured time.

The benchmarking methodology used in this section deviates from the starter benchmarking code in order to obtain stable and meaningful performance measurements. In particular, we revised the timing setup to:

- Separate warm-up iterations from steady-state measurements,
- Use explicit CUDA synchronization when benchmarking on GPU,
- Exclude timing utilities themselves from the compiled region, and
- Ensure inference-only execution using `model.eval()` and `torch.no_grad()`.

These design choices follow the recommendations and methodology outlined in the official PyTorch *torch.compile End-to-End Tutorial* [1], which explicitly notes that:
(i) the first few iterations of a compiled model are expected to be slower due to compilation overhead, and
(ii) speedups are hardware-dependent and may be smaller on non-datacenter GPUs.

Our revised benchmarking functions therefore align with PyTorch’s recommended practice for evaluating `torch.compile`, and avoid misleading conclusions caused by measurement overhead or compilation warm-up effects.


#### Results

| Device | Execution Mode | Runtime (s) | Speedup |
|------|----------------|-------------|----------|
| CPU  | Eager          | 2.86        | 1.00×    |
| CPU  | Compiled       | 1.90        | 1.50×    |
| GPU  | Eager          | 0.171       | 1.00×    |
| GPU  | Compiled       | 0.133       | 1.29×    |

#### Observations (Deep Learning Systems perspective)

- `torch.compile` introduces a cold-start latency due to graph capture and kernel compilation, which must be amortised over repeated executions.
- Speedups on consumer GPUs are smaller than those reported in datacenter settings due to hardware constraints.
- Correct benchmarking methodology is critical; performance conclusions can change significantly if warm-up, synchronization, and timing overhead are not handled carefully.


---

### Part B – Kernel Fusion: Naive vs Fused SDPA

#### Background

Kernel fusion is a performance optimisation technique that reduces overhead by collapsing multiple logical operators into a single kernel. This reduces both kernel launch costs and, more importantly, global memory traffic. The benefits of kernel fusion are most pronounced in memory-bound workloads, where performance is limited by memory bandwidth rather than arithmetic throughput.

Scaled Dot-Product Attention (SDPA) is a canonical example. In its naive formulation, attention is decomposed into a sequence of operations:

1. A matrix multiplication to compute attention scores (QKᵀ),
2. A scaling operation,
3. A softmax over the attention scores, and
4. A second matrix multiplication to compute the weighted sum with V.

Each of these operations is typically implemented as a separate kernel. Intermediate results, including the full attention matrix, are materialised in global memory and subsequently reloaded by later kernels. This leads to excessive global memory reads and writes, and makes the computation memory-bandwidth bound.

PyTorch provides a fused SDPA implementation via `torch.nn.functional.scaled_dot_product_attention`, which is based on the FlashAttention algorithm. Rather than accelerating individual operators, FlashAttention reformulates the attention computation to avoid materialising the full attention matrix altogether.

#### Profiling Results

| Device | Implementation | Dominant Kernel | Total Runtime |
|------|----------------|-----------------|----------------|
| CPU  | Naive SDPA     | `aten::bmm`     | ~2.38 s        |
| CPU  | Fused SDPA     | FlashAttention (CPU) | ~70.7 ms |
| CUDA | Naive SDPA     | `aten::bmm` + softmax | ~6.49 ms |
| CUDA | Fused SDPA     | FlashAttention (CUDA) | ~2.21 ms |

#### Observations

- On CPU, the fused SDPA implementation yields approximately a 33× speedup over the naive formulation.
- On GPU, the fused implementation achieves an approximate 3× speedup.
- In both cases, the naive implementation is dominated by matrix multiplication and softmax kernels, while the fused path executes a single attention kernel.
- The magnitude of the speedup differs between CPU and GPU, but the qualitative behaviour is consistent across devices.

#### Why is the speedup larger on CPU than on GPU?

Although kernel fusion improves performance on both CPU and GPU, the relative speedup is substantially larger on CPU. This difference arises from how naive SDPA is executed on each architecture.

On CPU, the naive SDPA implementation executes as a sequence of independent high-level operators, each of which:
- Materialises large intermediate tensors in main memory,
- Performs full memory traversals for matrix multiplication and softmax, and
- Incurs repeated cache misses due to limited reuse of intermediate results.

As a result, the naive CPU implementation is strongly memory-bandwidth bound and suffers from poor cache locality. Kernel fusion eliminates most intermediate tensor materialisation and reduces memory traffic, leading to a dramatic reduction in execution time.

On GPU, the naive implementation is already partially optimised:
- Matrix multiplications are executed using highly optimised cuBLAS kernels,
- Softmax kernels are reasonably efficient, and
- GPU memory bandwidth is significantly higher than CPU memory bandwidth.

Although the naive GPU implementation still incurs multiple kernel launches and global memory reads, these overheads represent a smaller fraction of total runtime compared to CPU. Consequently, kernel fusion yields a smaller relative speedup on GPU, even though the absolute runtime improvement remains significant.

In summary, kernel fusion removes a larger proportion of inefficiency in the CPU execution path than in the GPU path. The GPU already benefits from specialised compute kernels and high bandwidth, whereas the CPU naive implementation exposes more opportunities for optimisation through fusion.


#### Systems Interpretation

The performance gains observed are primarily due to reduced memory traffic rather than faster arithmetic. In the naive SDPA implementation, the attention matrix is fully materialised in global memory, then repeatedly read and written by subsequent kernels. This results in high global memory bandwidth consumption and poor data locality.

FlashAttention avoids this bottleneck by computing attention in tiles. Query, key, and value blocks are streamed from global memory into on-chip storage (registers and shared memory), and partial dot products are accumulated incrementally. A numerically stable “online softmax” is applied during this process, allowing the attention output to be computed without ever forming the full attention matrix in memory.

By fusing the entire attention computation into a single kernel and operating on small tiles, FlashAttention:
- Eliminates intermediate tensor materialisation,
- Reduces global memory reads and writes,
- Increases arithmetic intensity, and
- Improves cache, shared memory, and register locality.

This experiment demonstrates that kernel fusion is not merely an implementation detail, but an algorithmic transformation. The largest performance gains arise from restructuring the computation to better match the memory hierarchy, rather than from general-purpose compiler optimisations alone.

---

### Part C – Custom Kernels and MXINT8 Quantisation

#### Motivation: Why Custom Kernels Matter for Quantisation

While PyTorch provides highly optimised kernels for common numerical formats such as FP32, FP16, and BF16, these kernels are designed for general-purpose floating-point execution. When models are quantised to custom numerical formats, such as MXINT, default kernels are no longer optimal.

To fully exploit the benefits of quantisation, custom kernels are required. These kernels can be designed to:
- Operate directly on the quantised representation,
- Avoid unnecessary format conversions,
- Exploit hardware-friendly data layouts, and
- Minimise memory traffic and instruction overhead.

In this lab, a custom MXINT8 dequantisation kernel is used to demonstrate how numerical format design and kernel implementation interact to improve performance and memory efficiency.

---

#### MXINT8 Format Recap

MXINT is a block-scaled numerical format that lies between floating-point and fixed-point representations. Instead of storing a separate exponent for every value, MXINT groups multiple values together and shares a single exponent across the group.

An MXINT vector consists of:
- One shared 8-bit exponent (biased by 127), and
- Multiple signed fixed-point mantissas.

This structure can be represented as:

```

Exp |- Mantissa 1
|- Mantissa 2
|- ...
|- Mantissa (group_size)

```

During dequantisation, each mantissa is scaled by the same exponent factor. This preserves dynamic range while significantly reducing storage overhead compared to standard floating-point formats.

---

#### Question: How does MXINT8 benefit custom hardware when both activations and weights are quantised?

When both activations and weights in a linear layer are quantised to MXINT8, the format provides several key advantages for custom hardware and specialised execution kernels.
In a linear layer, where computation is dominated by large matrix multiplications, these advantages directly translate into lower memory traffic for multiply and accumulate operations and higher effective throughput.


#### 1. Reduced Memory Footprint and Bandwidth

MXINT8 significantly reduces the number of bits required per value compared to FP16 or FP32. Fewer bits per tensor element lead to:
- Lower memory bandwidth requirements,
- Improved cache utilisation, and
- Reduced pressure on global memory.

This is particularly beneficial for memory-bound workloads such as large matrix multiplications, where performance is often limited by data movement rather than compute.

#### 2. Simplified Arithmetic and Compute Units

MXINT mantissas are fixed-point values and can be processed using integer arithmetic. Because the exponent is shared across a group:
- Expensive per-element exponent handling is avoided,
- Arithmetic datapaths are simplified, and
- Hardware can use smaller, more energy-efficient compute units.

This enables higher throughput and lower power consumption compared to fully general-purpose floating-point execution.

#### 3. Hardware-Friendly Dataflow and Parallelism

The block structure of MXINT aligns naturally with SIMD, systolic array, and tensor-core-style architectures. A custom kernel or accelerator can:
- Load one exponent per group,
- Stream multiple mantissas through the same datapath, and
- Reuse scaling logic across many values.

This increases arithmetic intensity, improves data reuse, and reduces instruction overhead, all of which are critical for efficient accelerator design.

#### 4. Reduced General-Purpose Overhead (“Turing Tax”)

By fixing the numerical format and dataflow, MXINT8 enables hardware designs that move away from fully general-purpose execution. Compared to FP32 or FP16 execution:
- Instruction fetch and decode overhead is reduced,
- Control logic is simplified, and
- More silicon area can be devoted to useful computation.

This reduces the so-called *Turing Tax*: the performance and energy cost of using a universal programmable processor instead of a domain-specific accelerator.

---

### Role of the Custom MXINT8 Dequantisation Kernel

The custom MXINT8 dequantisation kernel used in this lab illustrates how these hardware advantages are realised in practice:
- The shared exponent is loaded once per group,
- Mantissas are converted using simple fixed-point scaling,
- Intermediate representations are minimised, reducing memory traffic, and
- Dequantisation is tightly integrated into the computation pipeline.

Compared to naïve, element-wise dequantisation using floating-point kernels, this approach is both more memory-efficient and better aligned with accelerator-style execution. Although dequantisation is unavoidable when executing linear layers in higher precision, performing it in a block-wise and kernel-fused manner ensures that the overhead does not negate the benefits of quantisation.


---

### Summary

MXINT8 benefits custom hardware when both weights and activations are quantised because it enables:
- Compact data representation,
- Integer-dominant arithmetic,
- Efficient block-wise computation, and
- Hardware designs that minimise general-purpose execution overhead.

In combination with custom kernels, MXINT8 allows accelerators to achieve high performance and energy efficiency while maintaining acceptable numerical accuracy. This reinforces a central theme of this lab: the largest performance gains arise not only from compiler optimisations, but from co-designing numerical formats, kernels, and hardware execution models.

---

## Part D – MXINT8 Dequantisation Kernel

### Kernel overview

The MXINT8 dequantisation kernel implements a weight-only quantisation workflow in which persistent weights are stored in a compact MXINT8 representation and expanded only when needed:

1. **Load** MXINT8 mantissas (`int8`) and shared micro-exponents (`uint8`) from global memory.
2. **Dequantise in-kernel** by reconstructing BF16 values using bit-level packing and a small correction step.
3. **Store** dequantised BF16 weights back to global memory for subsequent computation (e.g., GEMM).

This design reduces persistent memory footprint (MXINT8 storage) while retaining higher-precision arithmetic during compute. After the layer finishes, the temporary BF16 weights can be discarded.

---

### Question: Purpose of `dont_need_abs` and `bias`

The host reference implementation (mirrored by the device kernel) reconstructs BF16 values by packing sign, exponent, and fraction fields:

- **Sign bit** comes from the MXINT mantissa sign.
- **Exponent** comes from the shared micro-exponent (`scale`) for the group.
- **Fraction** comes from the lower bits of the mantissa magnitude.

A simplified view of the reconstruction logic is:

- `out`: BF16 bit-pattern composed from `(sign | exponent | fraction)`
- `bias`: BF16 value composed from `(sign | exponent | 0)` (same sign and exponent, zero fraction)

MXINT mantissas are not IEEE-normalised floats and therefore do not have an implicit leading 1. To represent a wider signed magnitude range using limited mantissa bits, MXINT uses a **region selector bit** (the `0x40` bit in the mantissa magnitude). This bit determines which decoding rule should be applied:

- If the region selector bit is set (`mantissa_abs & 0x40 != 0`), the packed value `out` already represents the intended magnitude and can be used directly.
- If the region selector bit is not set, the packed value must be offset-corrected by subtracting `bias`.

Formally, the kernel implements:

- **Rule A (no correction):** `y = out`
- **Rule B (offset correction):** `y = out - bias`

Here, `dont_need_abs` (or equivalently `dont_need_bias`) is the predicate selecting between these two decoding rules, and `bias` provides the baseline constant for the exponent bucket. This mechanism increases the effective representable range without increasing mantissa width.

---

### Question: How does `cta_tiler` partition data for copying? (CUTE `local_tile`)

In the CUDA kernel, the 1D mantissa array is reshaped into a logical 2D matrix by grouping elements according to `group_size`:

- Let `M = group_size`
- Let `K = num_groups`

After flattening, the mantissas are treated as a matrix of shape `(M, K)` where:
- The **row dimension** corresponds to the element index within a group.
- The **column dimension** corresponds to the group index (and thus the shared exponent index).

The CTA (thread block) uses a tiler:
- `cta_tiler = (BLK_M, BLK_K)`
and a block coordinate:
- `cta_coord = (blockIdx.x, blockIdx.y)`

`local_tile(mX, cta_tiler, cta_coord)` selects the rectangular sub-tensor owned by the CTA:

- Rows: `blockIdx.x * BLK_M ... blockIdx.x * BLK_M + BLK_M - 1`
- Cols: `blockIdx.y * BLK_K ... blockIdx.y * BLK_K + BLK_K - 1`

Thus, the global `(M, K)` matrix is decomposed into `(BLK_M × BLK_K)` tiles, each assigned to one CTA. The output tensor `mY` is tiled identically.

The shared exponent vector is tiled only along the group dimension (K). Each CTA loads the `BLK_K` exponents associated with the group-columns it processes, ensuring that all elements in a given column share the same exponent during reconstruction.

---

### Question: How does `layout_sX` partition threads for computation? (CUTE `local_partition`)

Within a CTA, CUTE uses a thread layout to map threads to elements of the CTA tile. The kernel constructs a 2D thread layout:

- `layout_tX = (thd_m, thd_k)`
- `dimBlock = thd_m * thd_k`

In the configuration used by the kernel, `thd_m = BLK_M` and `thd_k = BLK_K`, so the block contains `BLK_M * BLK_K` threads. This makes the thread layout isomorphic to the tile shape: each thread corresponds to a unique logical `(m, k)` position in the CTA tile.

CUTE then creates per-thread views using `local_partition`:
- `tXgX = local_partition(gX, layout_tX, threadIdx.x)` (thread view of global tile)
- `tXsX = local_partition(sX, layout_sX, threadIdx.x)` (thread view of shared-memory tile)

This mapping enables:
- **Elementwise cooperative loading**: each thread loads its assigned element from global memory into shared memory (guarded by predication on boundary tiles).
- **Elementwise reconstruction**: each thread reconstructs BF16 values for its assigned element(s), reusing the shared exponent corresponding to the tile column.
- **Elementwise cooperative storing**: each thread writes its reconstructed output back to global memory (again predicated on bounds).

Predication is implemented by partitioning an identity tensor through the same layout and comparing coordinates against the valid tile extents. This ensures that partial tiles at tensor boundaries do not issue out-of-bounds global memory accesses.

A key property is exponent reuse: the exponent index is derived from the K-coordinate (group column). Because threads are laid out across `(m, k)`, all threads with the same `k` (column) reuse the same shared exponent, matching MXINT’s group semantics.

---

### Summary

The MXINT8 dequantisation kernel combines:
- **Bit-level reconstruction** (sign/exponent/fraction packing plus a region-based bias correction),
- **Block tiling at the CTA level** (`cta_tiler` via `local_tile`), and
- **Thread-to-element mapping within a tile** (`layout_*` via `local_partition`),

to minimise global memory traffic, maximise exponent reuse, and ensure correct boundary handling via predication. This illustrates how numerical format design (shared micro-exponents) and GPU execution strategy (tiling + cooperative copy) co-determine performance in custom low-level kernels.

---

## Part E – Empirical Evaluation

This section evaluates the MXINT8 dequantisation kernel empirically, focusing on:
(i) latency characteristics on CPU versus GPU, and
(ii) realised GPU memory savings when MXINT8 is applied to a real Transformer model.

---

### Latency Profiling: CPU vs GPU Dequantisation

We benchmarked the MXINT8 dequantisation kernel using the provided test suite
(`test_ext_dequantize1d_latency`), which compares the host reference
implementation against the CUDA kernel across a range of tensor sizes and
group sizes.

#### Observed performance regimes

| Tensor size | Relative performance | Interpretation |
|------------|----------------------|----------------|
| Small      | GPU slower than CPU  | Kernel launch and synchronization overhead dominate |
| Large      | GPU significantly faster | Memory throughput and bandwidth dominate |

For small tensors (e.g. \(m = 1024\)), GPU execution is slower than the CPU
implementation. In this regime, the fixed cost of kernel launch, synchronization,
and dispatch outweighs any benefit from parallel execution.

As tensor size increases (e.g. \(m \ge 2{,}097{,}152\)), the GPU implementation
becomes orders of magnitude faster than the CPU version. In this regime, kernel
launch overhead is amortised, and performance is dominated by sustained memory
bandwidth rather than arithmetic complexity.

#### Interpretation

The MXINT8 dequantisation kernel is primarily **memory-bandwidth bound**. Once
sufficient data is available to saturate device memory throughput, the GPU’s
parallel memory system enables substantially higher throughput than the CPU.
Group size has a secondary effect, as the kernel remains dominated by global
memory traffic rather than computation.

---

### GPU Memory Savings in a Real Model

To evaluate the practical impact of MXINT8 quantisation, we applied the
`QLinearPacked` layer to a real Transformer-based emotion classification model
and measured peak GPU memory usage during inference.

**Experimental setup**
- GPU: NVIDIA GeForce RTX 2050 (4 GB)
- Model: `AnkitAI/deberta-v3-small-base-emotions-classifier`
- Precision: FP32 baseline vs MXINT8 weight-only quantisation
- Mode: `eval()` with `torch.no_grad()`

#### Peak memory usage

| Model variant | Peak GPU memory | Reduction |
|--------------|------------------|------------|
| FP32         | 559.94 MB        | —          |
| MXINT8       | 445.50 MB        | 20.4%      |

Both models produced identical predicted labels and very similar top-3 logits,
indicating that MXINT8 compression did not materially affect inference accuracy
for this example.

---

### Question: Why is the observed memory saving not the theoretical 74.2%?

The commonly cited theoretical reduction for MXINT8 weight storage,

$\frac{32 - (8 + 8/32)}{32} = 74.2\%$

represents an **upper bound** under idealised assumptions. In practice, peak GPU
memory usage reflects far more than persistent weight storage. The observed
reduction (~20%) is lower due to several factors:

1. **Activations and intermediate tensors remain high precision**  
   Even during inference, embeddings, attention intermediates, and MLP
   activations are allocated in FP16/FP32. These tensors dominate peak memory
   usage and are unaffected by weight-only quantisation.

2. **Not all parameters are quantised**  
   Only `Linear` layers are replaced with MXINT8 variants. Components such as
   embeddings, LayerNorms, biases, and the classifier head remain unquantised and
   continue to consume FP32 memory.

3. **MXINT storage introduces real overheads**  
   Shared exponents must be stored explicitly, and packed layouts introduce
   alignment and padding overheads that reduce the effective compression ratio.

4. **CUDA allocator behaviour affects peak measurements**  
   PyTorch reports peak allocated memory, which includes allocator caching,
   fragmentation, and temporary buffers. Peak memory does not scale linearly with
   parameter size alone.

#### Interpretation

The theoretical 74.2% reduction applies only to isolated weight storage.
In a realistic inference workload, peak GPU memory reflects activations,
temporary buffers, unquantised layers, and allocator effects. Consequently, the
observed ~20% reduction is expected and still represents a meaningful memory
saving achieved without changing model predictions.

---

### Conclusion

This lab demonstrates that the most substantial performance improvements in deep
learning systems arise from **algorithmic restructuring and hardware-aware
design**, rather than compiler automation alone.

- `torch.compile` provides convenient steady-state speedups but introduces
  unavoidable cold-start overhead.
- Kernel fusion (e.g. FlashAttention) reduces memory traffic by eliminating
  intermediate tensor materialisation.
- Custom kernels are essential for extracting the full benefit of specialised
  numerical formats such as MXINT8.
- Peak GPU memory usage reflects the entire execution context, not just parameter
  storage.

Accurate benchmarking and systems-level reasoning are therefore essential to
correctly interpret performance results.

---

### Key Takeaways

- Compilation overhead must be amortised to realise benefits from
  `torch.compile`.
- Kernel fusion improves performance primarily by reducing memory movement.
- Specialised numerical formats require custom kernels to be effective.
- Theoretical compression ratios overestimate real peak memory savings.
- Performance optimisation must be evaluated in realistic system contexts.




---


#### References

[1] PyTorch Documentation. *torch.compile End-to-End Tutorial*.  
https://docs.pytorch.org/tutorials/intermediate/torch_compile_full_example.html
