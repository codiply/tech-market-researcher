# Models

Gateway models are exposed as endpoints of the MLflow AI Gateway, provisioned by the bootstrapping script from `config/bootstrapping/main.yaml`.

Models on the local [OMLX](https://omlx.ai/) server are identified by the repository name **without the organization prefix** — OMLX strips it (e.g. the `mlx-community/embeddinggemma-2-8bit` download is served as `embeddinggemma-2-8bit`).

## Chat models

| Gateway endpoint | Model id | Notes |
| --- | --- | --- |
| `omlx.Qwen3.8-27B-oQ4e-mtp` | `Qwen3.8-27B-oQ4e-mtp` | Default agent model ([Jundot/Qwen3.8-27B-oQ4e-mtp](https://huggingface.co/Jundot/Qwen3.8-27B-oQ4e-mtp), [Qwen 3.8](https://huggingface.co/Qwen)) |
| `omlx.Qwen3.6-35B-A3B-oQ4e-mtp` | `Qwen3.6-35B-A3B-oQ4e-mtp` | [Jundot/Qwen3.6-35B-A3B-oQ4e-mtp](https://huggingface.co/Jundot/Qwen3.6-35B-A3B-oQ4e-mtp), [Qwen 3.6](https://huggingface.co/Qwen) |
| `bedrock.eu.amazon.nova-micro-v1` | `eu.amazon.nova-micro-v1:0` | [Nova Micro model card](https://docs.aws.amazon.com/bedrock/latest/userguide/model-card-amazon-nova-micro.html) on [Bedrock](https://aws.amazon.com/bedrock/) |
| `bedrock.eu.amazon.nova-2-lite-v1` | `eu.amazon.nova-2-lite-v1:0` | [Nova 2 Lite model card](https://docs.aws.amazon.com/bedrock/latest/userguide/model-card-amazon-nova-2-lite.html) on [Bedrock](https://aws.amazon.com/bedrock/) |
| `bedrock.eu.amazon.nova-pro-v1` | `eu.amazon.nova-pro-v1:0` | [Nova Pro model card](https://docs.aws.amazon.com/bedrock/latest/userguide/model-card-amazon-nova-pro.html) on [Bedrock](https://aws.amazon.com/bedrock/) |
| `ec2.Qwen3-4B` | `Qwen/Qwen3-4B` | [Qwen/Qwen3-4B](https://huggingface.co/Qwen/Qwen3-4B) on the [EC2 inference server](#ec2-inference-server-models) |
| `ec2.Qwen3.8-27B-FP8` | `Qwen/Qwen3.8-27B-FP8` | Official [Qwen FP8](https://huggingface.co/Qwen/Qwen3.8-27B-FP8) checkpoint on the [EC2 inference server](#ec2-inference-server-models) |
| `ec2.Qwen3.8-27B-NVFP4-QAD` | `nvidia/Qwen3.8-27B-NVFP4-QAD` | NVIDIA [NVFP4](https://huggingface.co/nvidia/Qwen3.8-27B-NVFP4-QAD) (Model Optimizer) checkpoint on the [EC2 inference server](#ec2-inference-server-models) |

## Embedding models

| Gateway endpoint | Model id | Notes |
| --- | --- | --- |
| `omlx.embeddinggemma-2-4bit` | `embeddinggemma-2-4bit` | [mlx-community/embeddinggemma-2-4bit](https://huggingface.co/mlx-community/embeddinggemma-2-4bit) quantization of [Google EmbeddingGemma 2](https://huggingface.co/google/embeddinggemma-2) |
| `omlx.Qwen3-Embedding-0.6B-4bit-DWQ` | `Qwen3-Embedding-0.6B-4bit-DWQ` | [mlx-community/Qwen3-Embedding-0.6B-4bit-DWQ](https://huggingface.co/mlx-community/Qwen3-Embedding-0.6B-4bit-DWQ) quantization of [Qwen3-Embedding-0.6B](https://huggingface.co/Qwen/Qwen3-Embedding-0.6B) |

## EC2 inference server models

The following models are served by [vLLM](https://docs.vllm.ai/) on the [EC2 inference server](../assets/infrastructure/cloudformation/inference-server/README.md), reachable over the tailnet. They can be reached two ways:

- **Directly**, by setting the agent's model provider to `ec2_inference` (e.g. through one of the `ec2-inference-*` mixins); the base URL comes from `EC2_INFERENCE_BASE_URL`.
- **Through the MLflow AI Gateway**, under the `ec2.`-prefixed endpoint names in the table above (also usable as the evals' judge model through the `ec2-inference-*` eval mixins).

| Model id | Notes |
| --- | --- |
| `Qwen/Qwen3-4B` | Small model, for g4dn (T4) instances |
| `Qwen/Qwen3.8-27B-FP8` | Official [Qwen FP8](https://huggingface.co/Qwen) checkpoint |
| `nvidia/Qwen3.8-27B-NVFP4-QAD` | NVIDIA [NVFP4](https://huggingface.co/nvidia/Qwen3.8-27B-NVFP4-QAD) (Model Optimizer) checkpoint, served with `--quantization modelopt` |

The model id is the Hugging Face checkpoint name passed to `vllm serve` (the serve scripts do not set a custom `--served-model-name`). The gateway connection needs no API key, so the evals and the direct provider share the same `EC2_INFERENCE_BASE_URL` env var.
