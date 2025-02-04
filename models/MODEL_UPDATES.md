# Model Updates

> [!NOTE]
>
> Please refer to the front-page [README](../README.md) for the latest verified release for each model.

## January 13, 2025

### [Llama 3.1/3.2](demos/llama3)
- Integrated Llama3 models (1B/3B/8B/11B/70B) into [vLLM fork](https://github.com/tenstorrent/vllm/tree/dev/tt_metal) for all compatible Tenstorrent devices (N150/N300/T3000/Galaxy).
- Enabled prefill with the maximum context length (131072) when running the Llama3 text models on smaller devices (N150/N300) via chunked prefill.

## December 16, 2024

### [Llama 3.1/3.2](demos/llama3)
- Added support for batch size 32 and the maximum context length (131072 tokens).
- Added full hardware compatibilty for the 1B/3B/8B/11B/70B models (all models are now compatible with N150, N300, QuietBox, Galaxy except for 70B which is only supported on QuietBox and Galaxy due to its large size).

## December 2, 2024

### [Llama 3.1/3.2](demos/llama3)
- Improved the decode performance of the 1B/3B/8B/11B text models (for 8B, increased from ~23 t/s/u to ~28 t/s/u) by using BFP4 weights (instead of BFP8) for FF1 and FF3 in the MLP.
- Added the option to specify custom model configurations, with two defaults for performance and accuracy already provided. 

## November 18, 2024

### [Llama 3.2 - 1B/3B/11B](demos/llama3)
- Created a new shared codebase for the Llama3 family of models, with newly added support for Llama3.2-1B/3B/11B.

### [Llama 3/3.1 - 70B](demos/t3000/llama3_70b)
- Added support for the `ttnn.experimental.rotary_embedding_llama` op in decode mode, eliminating unnecessary device transfers of rotation matrices.

## October 21, 2024

### [Llama 3/3.1 - 70B](demos/t3000/llama3_70b)
- Enabled prefill workloads to pad to multiples of 1024 instead of powers of 2, improving overall performance for longer sequences

## October 7, 2024

### [Llama 3.1 - 8B](demos/wormhole/llama31_8b)
- Added support for continuous batching
- Added paged caching support for PagedAttention
- Added a demo which runs with TT-NN tracing (23 t/s/u decode on main)

## September 23, 2024

### [Llama 3/3.1 - 70B](demos/t3000/llama3_70b)
- Added support for 128K context length using PagedAttention
- Added a continuous batching demo for running multiple batches of users consecutively
- Added the option to enable TT-NN tracing

## September 9, 2024

### [Mixtral7Bx8](demos/t3000/mixtral8x7b)
> **Note:** This feature is available as of release [v0.52.0-rc1](https://github.com/tenstorrent/tt-metal/tree/v0.52.0-rc1)
- Added support for any user prompt size up to a maximum of 32k tokens

## August 26, 2024

### [Falcon7B](demos/falcon7b_common)
- Added data parallel demo for a single Galaxy (32 chips)
- Refactored all modules and tests to use ttnn multi-device tensors

### [Llama 3.1 - 8B](demos/wormhole/llama31_8b)
> **Note:** This feature is available as of release [v0.51.0-rc33](https://github.com/tenstorrent/tt-metal/tree/v0.51.0-rc33)
- Added multi-batching support to the demo for running multiple batches of users consecutively

### [Mixtral7Bx8](demos/t3000/mixtral8x7b)
- Improved end-to-end performance through optimizations to the attention mask in flash decoding

## August 12, 2024

### [Llama 3.1 - 8B](demos/wormhole/llama31_8b)
- Added support for flash decoding

### [Mistral7B](demos/wormhole/mistral7b)
- Updated the demo to support multiple batches of users

### [Mamba-2.8B](demos/wormhole/mamba) 
- Updated the demo to use the full prefill graph instead of processing a single token of the prompt at a time using decode

### [Mixtral7Bx8](demos/t3000/mixtral8x7b)
- Added support for decode with 32K context length using flash decoding
- Fused mixture of experts into a single operation using `ttnn.moe`

## July 29, 2024

### [Llama 3.1 - 8B](demos/wormhole/llama31_8b)
- Added support for LLaMA 3.1 - 8B
- Runs fast prefill for sequence lengths of up to 512 tokens
- Supports a maximum context length of 8K tokens

### [Llama 3/3.1 - 70B](demos/t3000/llama3_70b)
- Added support for LLaMA 3.1 70B (new scaled rotary position embeddings)
- Prefill and decode now support 8K context length with batch size 16

### [Mistral7B](demos/wormhole/mistral7b)
- Added prefill support for 4K context length, using scaled dot product attention
