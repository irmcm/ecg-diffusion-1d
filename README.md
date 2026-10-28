# 1D Diffusion (DDPM) for Single-Lead ECG Synthesis

A PyTorch implementation of a 1D Denoising Diffusion Probabilistic Model (DDPM) trained on real single-lead ECG beats from the MIT-BIH Arrhythmia Database (Record 100, MLII lead, 360 Hz). 

This project explores generative deep learning for 1D physiological signals, featuring a leak-free evaluation pipeline and performance benchmarking against traditional linear and template baselines.

## Pipeline Overview

* **Leak-Free Chronological Split:** The 30-minute recording is split sequentially into 60% train, 10% validation, and 30% test sets. Beat windows are strictly prevented from crossing split boundaries.
* **Preprocessing:** Trained exclusively on normal sinus beats. Per-window median subtraction removes baseline wander, and amplitude scaling is fitted only on the training split.
* **Model Architecture:** A custom 1D U-Net (~0.59M parameters) using GroupNorm, SiLU activations, pre-activation residual blocks, sinusoidal time embeddings, and a cosine noise schedule with EMA weights.

## Key Design Notes & Findings

1. **Positional Alignment:** Standard 1D convolutions can cause peak drifting. Adding fixed coordinate channels and learned positional parameters gives the U-Net explicit spatial awareness, ensuring the R-peak stays precisely centered.
2. **Stable Sampling:** The reverse sampling process reconstructs $\hat{x}_0$ at each step and applies clipping to keep both the isoelectric baseline and peak amplitudes stable.
3. **Morphology vs. PCA:** While standard linear models (like PCA) capture overall global distribution well, the diffusion model better preserves sharp, non-linear local features—specifically capturing cleaner S-wave troughs and QRS widths without overly smoothing out high-frequency details.
4. **Anomaly Detection:** Reconstruction error across intermediate diffusion steps can also serve as an anomaly score to help flag premature beats (A/V arrhythmias) in held-out data.

## Limitations

* **Single Recording Scope:** Based on a single recording, one patient, and one lead; it does not claim cross-patient generalization.
* **Texture vs. Morphology:** Generated beats capture overall shape and beat-to-beat variability, but tend to be smoother than real sensor data, missing fine-scale high-frequency texture.
* **Clinical Status:** Synthetic beats are unassessed by clinicians and must not be used for medical diagnosis.

## Requirements & Usage
Install dependencies and run the notebook in Google Colab or locally:
```bash
pip install torch numpy scipy scikit-learn matplotlib wfdb pooch
