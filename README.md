# 🎙️ Wav2Vec2 Speech Recognition
### Feature Masking for Robust Low-Resource Classification

> *Does latent-space augmentation improve generalization in small-scale speech recognition? This project finds out.*

---

## Overview

This project implements a **speech command recognition system** built on a pretrained [Wav2Vec2](https://huggingface.co/facebook/wav2vec2-base) model, augmented with a lightweight **feature-level masking strategy** inspired by SpecAugment — applied in latent space rather than the raw spectrogram.

The system serves dual purposes: a **research platform** to rigorously evaluate augmentation strategies, and a **deployment-ready application** for interactive speech classification.

---

## Model Architecture

```
Raw Waveform (16 kHz)
        │
        ▼
┌─────────────────────┐
│   Wav2Vec2 Encoder  │  ← Pretrained transformer backbone
│  (Feature Extractor)│
└─────────────────────┘
        │
        ▼
┌─────────────────────┐
│  Attention Pooling  │  ← Mean-pooled latent embeddings
│  / Mean Pooling     │
└─────────────────────┘
        │
   [Training only]
        │
┌─────────────────────┐
│  Time Masking Layer │  ← Feature-level SpecAugment (latent space)
└─────────────────────┘
        │
        ▼
┌─────────────────────┐
│   FC Classification │  ← 10-class output head
│       Head          │
└─────────────────────┘
        │
        ▼
   Predicted Class
```

---

## Dataset

**Mini Speech Commands** — 10 classes, ~1 second per clip

| Split | Classes |
|-------|---------|
| `yes` `no` `up` `down` `left` | Training / Validation / Test |
| `right` `on` `off` `stop` `go` | Balanced across all splits |

- Audio resampled to **16 kHz**
- Balanced **train / validation / test** splits
- Minimal preprocessing — raw waveform fed directly to Wav2Vec2

---

## Experimental Design

Two configurations are evaluated head-to-head:

### Baseline
- Pretrained Wav2Vec2 encoder
- No augmentation
- Serves as the performance floor

### Proposed (Augmented)
- Wav2Vec2 + **feature-level time masking**
- Masking applied in the latent embedding space (post-encoder)
- Inspired by SpecAugment — but operating on learned representations, not spectrograms

### Research Question

> **Does feature-space masking improve generalization in small-scale speech classification?**

The hypothesis: by masking latent features during training, the model is forced to learn **contextual dependencies** across the full embedding rather than relying on local patterns — improving robustness on unseen samples.

---

## Evaluation

| Metric | Description |
|--------|-------------|
| **Accuracy** | Overall classification accuracy |
| **Per-class F1** | Precision/recall balance per command |
| **Confusion Matrix** | Misclassification heatmap |
| **Error Analysis** | Systematic misclassification patterns |

---

## System Features

### 🎤 Interactive Inference
- Upload any `.wav` file
- Real-time prediction with confidence scores
- Class probability distribution visualization

### 📊 Model Interpretation
- Per-class performance breakdown
- Confusion matrix analysis
- Baseline vs. augmented comparison dashboard

### 📄 Reporting
- Automatic PDF report generation
- Side-by-side system comparison summary

---

## Quickstart

```bash
# 1. Install dependencies
pip install -r requirements.txt

# 2. Launch the app
streamlit run app.py
```

---

## Requirements

```
torch
transformers
librosa
streamlit
plotly
seaborn
scikit-learn
```

---

## Tech Stack

| Layer | Technology |
|-------|-----------|
| Deep Learning | PyTorch |
| Speech Model | HuggingFace Transformers (Wav2Vec2) |
| Audio Processing | Librosa |
| Web Interface | Streamlit |
| Visualization | Plotly, Seaborn |
| Evaluation | scikit-learn |

---

## Key Insight

> Feature-level masking encourages the model to build **holistic representations** of speech commands — rather than relying on the presence of a single discriminative local feature. The result is a classifier that degrades more gracefully under noisy or partial inputs.

---

## Output

A complete end-to-end pipeline for:

- ✅ Speech command classification (10 classes)
- ✅ Rigorous model comparison (baseline vs. augmented)
- ✅ Interactive visualization and real-time inference
- ✅ Automated reporting and evaluation export

---

<div align="center">

*Built with PyTorch · Wav2Vec2 · Streamlit*

</div>
