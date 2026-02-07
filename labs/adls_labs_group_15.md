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

## Lab 4 – System Performance and torch.compile

### Setup

### Results

### Observations

### Conclusion

### Key Takeaways


---
