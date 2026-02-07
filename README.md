<div align="center">

# CogVideoX-experiments

**Text-to-Video Generation & Hyperparameter Ablation Studies**

[![Python 3.12+](https://img.shields.io/badge/Python-3.12%2B-3776AB?logo=python&logoColor=white)](https://python.org)
[![PyTorch](https://img.shields.io/badge/PyTorch-CUDA-EE4C2C?logo=pytorch&logoColor=white)](https://pytorch.org)
[![Hugging Face](https://img.shields.io/badge/HuggingFace-Diffusers-FFD21E?logo=huggingface&logoColor=black)](https://huggingface.co/THUDM/CogVideoX-2b)
[![License: MIT](https://img.shields.io/badge/License-MIT-green.svg)](LICENSE)

*Systematic exploration of CogVideoX-2b, a 2-billion parameter diffusion model by THUDM (Tsinghua University), through controlled ablation studies on generation hyperparameters.*

</div>

---

## Table of Contents

- [Overview](#overview)
- [Key Findings](#key-findings)
- [Architecture](#architecture)
- [Setup](#setup)
- [Usage](#usage)
- [Experiments](#experiments)
- [Repository Structure](#repository-structure)
- [Colab Notebook](#colab-notebook)
- [Experiment Log](#experiment-log)
- [Author](#author)

---

## Overview

This project investigates text-to-video generation using [CogVideoX-2b](https://huggingface.co/THUDM/CogVideoX-2b) with two primary objectives:

1. **Inference demonstration** across 9 diverse text prompts spanning realistic, fantastical, and cinematic scenarios
2. **Ablation studies** on 4 hyperparameters (`guidance_scale`, `num_inference_steps`, `seed`, `num_frames`), each tested in isolation on a fixed prompt to measure individual impact

All experiments follow a controlled methodology: one parameter is varied while all others are held at their default values, enabling direct causal attribution of observed changes.

---

## Key Findings

| # | Parameter | Values Tested | Key Observation |
|:-:|---|---|---|
| 1 | `guidance_scale` | 1, 6, 12 | GS=6 is the optimal sweet spot; GS=1 produces severe distortion; GS=12 yields sharper but artificial results |
| 2 | `num_inference_steps` | 10, 25, 50 | **10 steps = complete failure** (black blob on green noise); 25 steps = near-perfect quality at half the time |
| 3 | `seed` | 42, 123, 999 | Seeds produce genuinely different compositions (camera angle, lighting, layout), not just noise variations |
| 4 | `num_frames` | 25, 49 | Per-frame quality is identical; only video duration and VRAM usage change |

> **Most surprising discovery:** there is a critical threshold at ~20-25 inference steps below which the diffusion model completely fails to denoise. This non-linear behavior disproves the assumption that quality degrades proportionally with fewer steps.

---

## Architecture

```
                    ┌──────────────────────────────┐
                    │     prompts/*.txt             │  File-based prompt system
                    │  (zero code changes needed)   │  (just add a .txt file)
                    └──────────────┬───────────────┘
                                   │
                                   ▼
┌──────────────┐    ┌──────────────────────────────┐    ┌──────────────────┐
│  CLI args    │───▶│     run_cogvideox.py          │───▶│  outputs/*.mp4   │
│  --steps     │    │                              │    │                  │
│  --guidance  │    │  CogVideoXPipeline (HF)      │    │  Organized by    │
│  --seed      │    │  + CPU offload               │    │  experiment      │
│  --frames    │    │  + VAE tiling/slicing         │    │  subfolder       │
│  --output    │    │                              │    │                  │
└──────────────┘    └──────────────────────────────┘    └──────────────────┘
```

**Design principles:**
- **Modular prompts** — each prompt is a standalone `.txt` file; adding a new scenario requires zero code changes
- **CLI-driven experiments** — all hyperparameters are exposed as command-line arguments
- **Memory-optimized** — sequential CPU offload + VAE tiling/slicing enable generation on GPUs with ~15 GB VRAM
- **Reproducible** — fixed seeds guarantee identical outputs across runs

---

## Setup

### Prerequisites

| Requirement | Version |
|---|---|
| Python | 3.12+ |
| CUDA | 12.x |
| GPU VRAM | ~15 GB (with optimizations) |
| Platform | Linux / WSL2 (tested on Ubuntu) |

### Installation

```bash
git clone https://github.com/claudio-dragotta/CogVideoX-experiments.git
cd CogVideoX-experiments

python -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
```

The model weights (~4.5 GB) are automatically downloaded from Hugging Face on first run.

---

## Usage

### Single video generation

```bash
python run_cogvideox.py --prompt_file panda_guitar
```

### CLI parameters

| Parameter | Default | Description |
|:---|:---:|:---|
| `--prompt_file` | *(required)* | Name of the `.txt` file in `prompts/` (without extension) |
| `--model` | `2b` | Model variant: `2b` (float16) or `5b` (bfloat16) |
| `--steps` | `50` | Number of diffusion denoising steps |
| `--guidance` | `6.0` | Classifier-free guidance scale |
| `--frames` | `49` | Number of video frames to generate |
| `--fps` | `8` | Output video frame rate (playback speed) |
| `--seed` | `42` | Random seed for reproducibility |
| `--output` | *auto* | Custom output path (default: `outputs/<model>/<prompt>.mp4`) |

### Batch generation

```bash
# Generate all 9 baseline videos (default parameters)
./run_all.sh

# Generate all 20 experiment videos (baseline + ablation studies)
./run_experiments.sh
```

`run_experiments.sh` includes **skip logic**: existing videos are not regenerated, allowing safe interruption and resumption.

---

## Experiments

### Baseline — 9 prompts with default parameters

| Prompt | Scene Description | Category |
|:---|:---|:---:|
| `panda_guitar` | Panda playing acoustic guitar in a bamboo forest | Fantastical |
| `gatto_hacker` | Orange cat hacking on monitors, cyberpunk aesthetic | Fantastical |
| `samurai_pioggia` | Samurai unsheathing katana in rain on Japanese bridge | Cinematic |
| `astronauta_luna` | Astronaut on the Moon waving at Earth, NASA style | Cinematic |
| `ronaldo_pizza` | Cristiano Ronaldo eating pizza in Italian restaurant | Realistic |
| `persona_panino` | Person eating a sandwich with satisfaction, close-up | Realistic |
| `cane_parco` | Golden retriever jumping over agility hurdle in a park | Realistic |
| `citta_futuristica` | Futuristic city at night with flying cars, Blade Runner | Cinematic |
| `oceano_balena` | Whale breaching at sunset, National Geographic style | Cinematic |

### Ablation studies — output structure

All ablation experiments use `panda_guitar` as the fixed prompt, varying one parameter at a time:

```
outputs/
├── baseline/                    # 9 prompts × default params (steps=50, GS=6, seed=42, frames=49)
├── exp1_guidance_scale/         # GS = 1, 6, 12
├── exp2_steps/                  # steps = 10, 25, 50
├── exp3_seed/                   # seed = 42, 123, 999
└── exp4_frames/                 # frames = 25, 49
```

**Total: 20 generated videos** (9 baseline + 11 ablation variations)

### Impact summary

| Rank | Parameter | Affects | Time Impact | Recommended Setting |
|:---:|:---|:---|:---:|:---|
| 1 | `guidance_scale` | Visual quality & prompt adherence | None | **6** (default) |
| 2 | `num_inference_steps` | Detail quality & generation speed | Linear | **50** final / **25** preview |
| 3 | `seed` | Composition, camera angle, lighting | None | Try 3–5, pick best |
| 4 | `num_frames` | Video duration & VRAM usage | Proportional | **49** final / **25** preview |

---

## Repository Structure

```
CogVideoX-experiments/
├── run_cogvideox.py             # Main generation script (CLI interface)
├── run_cogvideox_colab.ipynb    # Self-contained Colab notebook
├── run_all.sh                   # Batch: generate all baseline videos
├── run_experiments.sh           # Batch: generate all experiment videos
├── requirements.txt             # Python dependencies
├── experiment_log.pdf           # Detailed experiment report (5 pages)
├── prompts/                     # Text prompt files (.txt)
│   ├── panda_guitar.txt
│   ├── gatto_hacker.txt
│   ├── samurai_pioggia.txt
│   ├── astronauta_luna.txt
│   ├── ronaldo_pizza.txt
│   ├── persona_panino.txt
│   ├── cane_parco.txt
│   ├── citta_futuristica.txt
│   └── oceano_balena.txt
└── outputs/                     # Generated videos (.mp4)
    ├── baseline/
    ├── exp1_guidance_scale/
    ├── exp2_steps/
    ├── exp3_seed/
    └── exp4_frames/
```

---

## Colab Notebook

The notebook [`run_cogvideox_colab.ipynb`](run_cogvideox_colab.ipynb) is fully self-contained and designed to run on **Google Colab** (T4 or A100 runtime). It includes:

- Environment setup and model loading with memory optimizations
- Baseline inference with a sample prompt
- 4 ablation experiments with side-by-side visual comparisons
- Key frame extraction and comparison grids
- Timing analysis and performance charts
- Summary of findings and conclusions

No local GPU required — all computation runs in the Colab runtime.

---

## Experiment Log

A detailed 5-page experiment report is available in [`experiment_log.pdf`](experiment_log.pdf). It covers:

1. **Objective** — project goals and methodology
2. **Setup** — hardware, software, and model configuration
3. **Pipeline Architecture** — file-based prompt system and CLI design
4. **Hyperparameter Impact Overview** — theoretical analysis of each parameter
5. **Experiments** — detailed visual observations for each ablation study
6. **Cross-Experiment Analysis** — comparative ranking and practical workflow
7. **Conclusions & Reflections** — key takeaways, limitations, and future directions

---

## References

- **CogVideoX** — Zhu et al., *CogVideoX: Text-to-Video Diffusion Models with An Expert Transformer*, 2024 ([arXiv:2408.06072](https://arxiv.org/abs/2408.06072))
- **Hugging Face Model Card** — [THUDM/CogVideoX-2b](https://huggingface.co/THUDM/CogVideoX-2b)
- **Diffusers Library** — [huggingface/diffusers](https://github.com/huggingface/diffusers)

---

## Author

**Claudio Dragotta**
- Email: [claudiodragotta@gmail.com](mailto:claudiodragotta@gmail.com)
- GitHub: [@claudio-dragotta](https://github.com/claudio-dragotta)
