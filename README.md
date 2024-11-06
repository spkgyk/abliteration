# Abliteration

This project is inspired by [FailSpy's abliterator](https://github.com/FailSpy/abliterator) but reimplemented without TransformerLens for improved speed and efficiency. I found the TransformerLens package both slow, memory hungry and limited -- only a few of the most common models were implemented. To bypass this, this repo uses PyTorch hooks to implement the same behaviour. 

## Installation

Requires conda for easy setup via:
```bash
source setup/setup.sh
```
Note: MacOS users should comment out `pytorch-cuda` and `flash-attn` dependencies. If not using conda, simply install the requirements found in the requirements.yaml file.

## Quick Start

See `abliterate.ipynb` for example usage.

## Overview

Abliteration is a technique that modifies large language model (LLM) behavior by identifying and manipulating specific directions in the model's activation space. The primary application is controlling refusal behavior - the tendency of models to reject certain types of prompts.

## How It Works

### The Science Behind Abliteration

Recent research has shown that refusal behavior in LLMs can often be traced to a single direction in the model's activation space (Arditi et al., 2024). This "refusal vector" consistently activates when the model encounters potentially problematic prompts. By manipulating this vector, we can control how the model responds to such inputs.

### Mathematical Framework

Let's examine the key components:

- $a$ represents the residual stream activation for a given input
- $\hat{r}$ represents the refusal direction vector
- $c_{\text{out}}$ represents a component's output before modification
- $c'_{\text{out}}$ represents the modified output

The core abliteration operation removes the projection onto the refusal direction:

```math
c'_{\text{out}} = c_{\text{out}} - (c_{\text{out}} \cdot \hat{r}) \hat{r}
```

### Implementation Approaches

Abliteration can be applied in two ways:

1. **Runtime Modification**
   - Dynamically adjusts activations during inference
   - Non-permanent changes
   - Useful for experimentation and testing

2. **Weight Modification**
   - Permanently modifies model weights
   - Orthogonalizes weights relative to the refusal direction
   - More efficient for production use


## Technical Details

### Computing the Refusal Vector

For a set of harmful prompts with activations $a_{\text{harmful}}^{(i)}$, we calculate the average projection:

```math
\text{avg\_proj}_{\text{harmful}} = \frac{1}{n} \sum_{i=1}^{n} (a_{\text{harmful}}^{(i)} \cdot \hat{r})
```

### Modifying Harmless Prompts

To force refusal on harmless inputs, we align their activations with the harmful average:

```math
a'_{\text{harmless}} = a_{\text{harmless}} - (a_{\text{harmless}} \cdot \hat{r}) \hat{r} + (\text{avg\_proj}_{\text{harmful}}) \hat{r}
```

## Acknowledgments

This project is a small contribution to the fantastic work below:

- [Refusal in LLMs is Mediated by a Single Direction](https://arxiv.org/abs/2406.11717) by Arditi et al. -- the original paper introducing abliteration
- [Refusal in LLMs is mediated by a single direction](https://www.lesswrong.com/posts/jGuXSZgv6qfdhMCuJ/refusal-in-llms-is-mediated-by-a-single-direction) -- an article by the original authors
- [FailSpy's abliterator](https://github.com/FailSpy/abliterator) - The original GitHub code that open-sourced this work
- [Maxime Labonne's Understanding Abliteration](https://huggingface.co/blog/mlabonne/abliteration) - An excellent explanation of the core concepts


## Contributing

Contributions are welcome! If you find any issues or have improvements to suggest, please submit a PR.

## License
