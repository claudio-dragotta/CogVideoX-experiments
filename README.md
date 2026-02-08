<div align="center">

# CogVideoX-experiments

### Systematic Hyperparameter Ablation Studies for Text-to-Video Diffusion

[![Python 3.12+](https://img.shields.io/badge/python-3.12+-blue.svg?style=for-the-badge&logo=python&logoColor=white)](https://python.org)
[![PyTorch](https://img.shields.io/badge/PyTorch-CUDA-EE4C2C?style=for-the-badge&logo=pytorch&logoColor=white)](https://pytorch.org)
[![Hugging Face](https://img.shields.io/badge/Diffusers-HuggingFace-FFD21E?style=for-the-badge&logo=huggingface&logoColor=black)](https://huggingface.co/THUDM/CogVideoX-2b)
[![License](https://img.shields.io/badge/License-MIT-green.svg?style=for-the-badge)](LICENSE)

<br>

A controlled experimental framework for investigating [CogVideoX-2b](https://huggingface.co/THUDM/CogVideoX-2b), a 2-billion parameter diffusion transformer by THUDM (Tsinghua University), through systematic ablation studies on generation hyperparameters.

<br>

[Quick Start](#quick-start) · [Key Findings](#key-findings) · [Experiments](#experiments) · [Documentation](#experiment-log)

</div>

<br>

## Highlights

- **Controlled Methodology** — Single-variable ablation: one parameter varies while all others remain constant
- **20 Generated Videos** — 9 baseline scenarios + 11 ablation variations across 4 hyperparameters
- **Memory Optimized** — Runs on ~15GB VRAM via sequential CPU offload + VAE tiling/slicing
- **Fully Reproducible** — Fixed seeds guarantee identical outputs across runs
- **Modular Design** — File-based prompts, CLI-driven experiments, zero code changes for new scenarios

<br>

## Quick Start

```bash
# Clone and setup
git clone https://github.com/claudio-dragotta/CogVideoX-experiments.git
cd CogVideoX-experiments
python -m venv .venv && source .venv/bin/activate
pip install -r requirements.txt

# Generate a single video
python run_cogvideox.py panda_guitar

# Run all experiments (baseline + ablation studies)
./run_experiments.sh
```

> Model weights (~4.5 GB) are downloaded automatically from Hugging Face on first run.

<br>

## Key Findings

| Parameter | Values Tested | Observation |
|:----------|:-------------:|:------------|
| **Guidance Scale** | 1, 6, 12 | `6` is optimal; `1` causes severe distortion; `12` yields sharper but artificial results |
| **Inference Steps** | 10, 25, 50 | Critical threshold at ~20 steps—below this, complete denoising failure occurs |
| **Seed** | 42, 123, 999 | Controls composition, camera angle, and lighting—not just noise variations |
| **Frame Count** | 25, 49 | Per-frame quality identical; affects only duration and VRAM usage |

<br>

<details>
<summary><b>Key Discovery: Non-linear Step Threshold</b></summary>

<br>

The most significant finding is a **critical threshold at ~20-25 inference steps** below which the diffusion model completely fails to denoise, producing black blobs on green noise. This non-linear behavior contradicts the assumption that quality degrades proportionally with fewer steps.

**Practical implication:** Use 25 steps for rapid previews (near-perfect quality at half the computation time) and 50 steps for final renders.

</details>

<br>

## Requirements

| Component | Specification |
|:----------|:--------------|
| Python | 3.12+ |
| CUDA | 12.x |
| GPU VRAM | ~15 GB (with memory optimizations enabled) |
| Platform | Linux / WSL2 |

### Dependencies

```
diffusers>=0.30.1
transformers>=4.44.2
accelerate>=0.33.0
imageio-ffmpeg>=0.5.1
```

<br>

## Usage

### Command Line Interface

```bash
python run_cogvideox.py <prompt_name> [options]
```

| Option | Default | Description |
|:-------|:-------:|:------------|
| `prompt_name` | *required* | Name of `.txt` file in `prompts/` (without extension) |
| `--model` | `2b` | Model variant: `2b` (float16) or `5b` (bfloat16) |
| `--steps` | `50` | Diffusion denoising steps |
| `--guidance` | `6.0` | Classifier-free guidance scale |
| `--frames` | `49` | Number of frames to generate |
| `--fps` | `8` | Output video frame rate |
| `--seed` | `42` | Random seed for reproducibility |
| `--output` | *auto* | Custom output path |

### Batch Scripts

```bash
./run_all.sh          # Generate 9 baseline videos
./run_experiments.sh  # Generate all 20 videos (includes skip logic for existing files)
```

<br>

## Experiments

### Baseline: 9 Diverse Scenarios

| Prompt | Description | Category |
|:-------|:------------|:--------:|
| `panda_guitar` | Panda playing acoustic guitar in bamboo forest | Fantastical |
| `gatto_hacker` | Orange cat hacking on monitors, cyberpunk aesthetic | Fantastical |
| `samurai_pioggia` | Samurai unsheathing katana in rain on Japanese bridge | Cinematic |
| `astronauta_luna` | Astronaut on the Moon waving at Earth, NASA style | Cinematic |
| `ronaldo_pizza` | Cristiano Ronaldo eating pizza in Italian restaurant | Realistic |
| `persona_panino` | Person eating a sandwich with satisfaction, close-up | Realistic |
| `cane_parco` | Golden retriever jumping over agility hurdle | Realistic |
| `citta_futuristica` | Futuristic city at night with flying cars | Cinematic |
| `oceano_balena` | Whale breaching at sunset, National Geographic style | Cinematic |

### Ablation Studies

All ablation experiments use `panda_guitar` as the fixed prompt:

```
outputs/
├── baseline/               # 9 prompts × default parameters
├── exp1_guidance_scale/    # guidance = 1, 6, 12
├── exp2_steps/             # steps = 10, 25, 50
├── exp3_seed/              # seed = 42, 123, 999
└── exp4_frames/            # frames = 25, 49
```

### Parameter Impact Summary

| Rank | Parameter | Affects | Recommended |
|:----:|:----------|:--------|:------------|
| 1 | `guidance_scale` | Visual quality, prompt adherence | **6** (default) |
| 2 | `num_inference_steps` | Detail quality, generation speed | **50** (final) / **25** (preview) |
| 3 | `seed` | Composition, camera angle, lighting | Try 3-5, select best |
| 4 | `num_frames` | Duration, VRAM usage | **49** (final) / **25** (preview) |

<br>

## Architecture

```
┌─────────────────────────────────────────────────────────────────────────┐
│                              PIPELINE                                    │
├─────────────────────────────────────────────────────────────────────────┤
│                                                                         │
│   prompts/*.txt ──────┐                                                 │
│   (file-based)        │                                                 │
│                       ▼                                                 │
│   CLI Arguments ──► run_cogvideox.py ──► outputs/<experiment>/*.mp4    │
│   --steps, --seed     │                                                 │
│   --guidance, etc.    │                                                 │
│                       ▼                                                 │
│              ┌────────────────────┐                                     │
│              │  CogVideoXPipeline │                                     │
│              │  + CPU offload     │                                     │
│              │  + VAE tiling      │                                     │
│              └────────────────────┘                                     │
│                                                                         │
└─────────────────────────────────────────────────────────────────────────┘
```

<br>

## Repository Structure

```
CogVideoX-experiments/
├── run_cogvideox.py            # Main generation script
├── run_cogvideox_colab.ipynb   # Self-contained Colab notebook
├── run_all.sh                  # Batch: all baseline videos
├── run_experiments.sh          # Batch: all experiments
├── requirements.txt            # Python dependencies
├── experiment_log.pdf          # Detailed experiment report
├── prompts/                    # Text prompts (.txt files)
│   ├── panda_guitar.txt
│   ├── gatto_hacker.txt
│   └── ... (9 total)
└── outputs/                    # Generated videos (.mp4)
    ├── baseline/
    ├── exp1_guidance_scale/
    ├── exp2_steps/
    ├── exp3_seed/
    └── exp4_frames/
```

<br>

## Colab Notebook

The notebook [`run_cogvideox_colab.ipynb`](run_cogvideox_colab.ipynb) runs entirely on **Google Colab** (T4/A100 runtime) with no local GPU required:

- Environment setup with memory optimizations
- Baseline inference + 4 ablation experiments
- Side-by-side visual comparisons and key frame extraction
- Timing analysis and performance charts

<br>

## Experiment Log

A comprehensive experiment report is available in [`experiment_log.pdf`](experiment_log.pdf):

1. Objectives and methodology
2. Hardware/software configuration
3. Pipeline architecture details
4. Hyperparameter impact analysis
5. Visual observations for each ablation
6. Cross-experiment comparative ranking
7. Conclusions, limitations, and future directions

<br>

## References

- **CogVideoX** — Zhu et al., *CogVideoX: Text-to-Video Diffusion Models with An Expert Transformer*, 2024 ([arXiv:2408.06072](https://arxiv.org/abs/2408.06072))
- **Model** — [THUDM/CogVideoX-2b](https://huggingface.co/THUDM/CogVideoX-2b) on Hugging Face
- **Framework** — [Diffusers](https://github.com/huggingface/diffusers) by Hugging Face

<br>

---

<div align="center">

**Author:** Claudio Dragotta · [claudiodragotta@gmail.com](mailto:claudiodragotta@gmail.com) · [@claudio-dragotta](https://github.com/claudio-dragotta)

</div>
