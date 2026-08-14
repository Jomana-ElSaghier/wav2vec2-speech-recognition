# Wav2Vec2 Speech Recognition

## Feature Masking for Robust Low-Resource Speech Classification

> Does latent-space augmentation improve generalization in small-scale speech recognition?

This project implements a speech command recognition system based on a pretrained **Wav2Vec2** model and investigates whether applying **feature-level time masking in latent space** can improve generalization in a low-resource speech classification setting.

The project combines a controlled machine learning experiment with an interactive **Streamlit application** for model evaluation, visualization, comparison, and automated reporting.

---

## Overview

Speech recognition models can benefit significantly from data augmentation, particularly when training data is limited. Traditional SpecAugment techniques typically operate on acoustic representations such as spectrograms.

In this project, a lightweight masking strategy inspired by SpecAugment is applied directly to the **latent representations produced by Wav2Vec2**.

Two configurations are evaluated:

- **Baseline:** Pretrained Wav2Vec2 with no feature-level augmentation.
- **Augmented:** Wav2Vec2 with latent-space time masking during training.

The objective is to determine whether masking learned speech representations can encourage the classifier to rely on more distributed information and improve performance on unseen samples.

---

## Research Question

> **Does feature-space masking improve generalization in small-scale speech classification?**

### Hypothesis

Applying time masking to latent speech representations may reduce the model's dependence on individual local features and encourage it to learn more robust contextual representations of speech commands.

---

## Model Architecture

### Baseline

```text
Raw Waveform (16 kHz)
        |
        v
+----------------------+
|      Wav2Vec2        |
|   Pretrained Encoder |
+----------------------+
        |
        v
 Hidden Representations
        |
        v
    Mean Pooling
        |
        v
+----------------------+
|  Classification Head |
|       10 Classes      |
+----------------------+
        |
        v
   Predicted Command
```

### Augmented Model

```text
Raw Waveform (16 kHz)
        |
        v
+----------------------+
|      Wav2Vec2        |
|   Pretrained Encoder |
+----------------------+
        |
        v
 Hidden Representations
        |
        v
+----------------------+
|   Latent Time        |
|      Masking         |
+----------------------+
        |
        v
    Mean Pooling
        |
        v
+----------------------+
|  Classification Head |
|       10 Classes      |
+----------------------+
        |
        v
   Predicted Command
```

The key difference between the two configurations is the application of **time masking to the learned Wav2Vec2 representations during training**.

---

## Dataset

The project uses the **Mini Speech Commands** dataset for 10-class speech command classification.

The target classes are:

```text
yes
no
up
down
left
right
on
off
stop
go
```

The dataset consists of short speech command recordings of approximately one second.

### Preprocessing

- Audio is resampled to **16 kHz**.
- Audio is processed as a raw waveform before being passed to Wav2Vec2.
- The classification task contains **10 speech command classes**.
- Training, validation, and test sets are used for model development and evaluation.

---

## Experimental Design

The experiment compares two models under the same classification task.

### Baseline Model

The baseline consists of:

- Pretrained Wav2Vec2 encoder
- No augmentation
- Mean pooling of latent representations
- Fully connected classification head
- 10 output classes

The baseline establishes the reference performance against which the proposed approach is evaluated.

### Augmented Model

The proposed configuration consists of:

- Pretrained Wav2Vec2 encoder
- Latent-space time masking
- Mean pooling
- Fully connected classification head
- 10 output classes

The masking is applied to the hidden representations after the Wav2Vec2 encoder rather than directly to the raw waveform or spectrogram.

---

## Evaluation

The models are evaluated using multiple metrics and visual analyses.

| Metric | Description |
|---|---|
| Accuracy | Overall classification performance |
| Per-class Performance | Performance breakdown for each speech command |
| Confusion Matrix | Visualization of correct and incorrect predictions |
| Error Analysis | Investigation of systematic misclassification patterns |
| Training Curves | Comparison of training behavior between models |

The evaluation is designed to compare not only overall accuracy but also how the augmentation affects individual speech command classes.

---

## Results

The experiment produced the following test-set accuracy:

| Model | Test Accuracy |
|---|---:|
| Baseline | 92% |
| Augmented | 98% |
| Improvement | +6 percentage points |

The augmented model achieved higher test accuracy than the baseline in the evaluated experiment.

The Streamlit dashboard also provides per-class analysis and confusion-matrix visualization to examine where the improvement occurs.

> **Note:** The test set is relatively small, so these results should be interpreted within the scope of this experimental setup rather than as a general benchmark for speech recognition.

---

## System Features

### Interactive Inference

The Streamlit application provides an interface for testing trained models with `.wav` audio files.

Features include:

- WAV audio upload
- Audio waveform visualization
- Log-spectrogram visualization
- Baseline model prediction
- Augmented model prediction
- Prediction confidence visualization
- Class probability distribution

### Model Evaluation

The application provides:

- Baseline versus augmented performance comparison
- Test accuracy comparison
- Per-class performance analysis
- Confusion matrix visualization
- Training loss curves
- Training accuracy curves
- Error analysis

### Automated Reporting

The application includes automated PDF report generation containing:

- Model performance comparison
- Architecture summary
- Dataset information
- Per-class performance
- Experimental conclusion

---

## Streamlit Application

The project includes an interactive dashboard built with Streamlit.

The application is organized into the following sections:

### Test Speech

Upload a WAV file and compare predictions from the baseline and augmented models.

### Training Curves

Visualize training loss and accuracy for both models across the training epochs.

### Results Overview

Explore:

- Test accuracy
- Model improvement
- Confusion matrix
- Per-class performance
- Error analysis

### PDF Report

Generate an automated summary report containing the main experimental results.

### Project Info

View the project objective, dataset, architecture, training configuration, and key findings.

---

## Project Structure

```text
wav2vec2-speech-recognition/
|
|-- app.py
|-- model.py
|-- utils.py
|-- requirements.txt
|-- README.md
|-- Speech Project.ipynb
|-- .gitignore
|-- .gitattributes
|
|-- models/
|   |-- baseline_model.pt
|   |-- augmented_model.pt
|   `-- README.md
|
```

### Main Files

| File | Description |
|---|---|
| `app.py` | Streamlit application and interactive dashboard |
| `model.py` | Wav2Vec2 model architecture |
| `utils.py` | Supporting utilities, preprocessing, inference, and reporting |
| `Speech Project.ipynb` | Model development, training, and experimental analysis |
| `requirements.txt` | Python dependencies |
| `models/` | Local directory for trained model checkpoints |

The trained model checkpoints are not included in the public repository because of their size. They should be placed locally in the `models/` directory.

---

## Installation

### 1. Clone the Repository

```bash
git clone https://github.com/YOUR_USERNAME/wav2vec2-speech-recognition.git
cd wav2vec2-speech-recognition
```

### 2. Install Dependencies

```bash
pip install -r requirements.txt
```

### 3. Add the Trained Models

Place the trained model checkpoints inside:

```text
models/
|
|-- baseline_model.pt
`-- augmented_model.pt
```

The Streamlit application expects these files at the above locations.

### 4. Run the Application

```bash
python -m streamlit run app.py
```

The application will be available locally through the Streamlit server.

---

## Requirements

The main dependencies include:

```text
torch
transformers
librosa
numpy
pandas
streamlit
plotly
matplotlib
seaborn
scikit-learn
soundfile
reportlab
```

---

## Technology Stack

| Layer | Technology |
|---|---|
| Programming Language | Python |
| Deep Learning | PyTorch |
| Speech Representation | Wav2Vec2 |
| Transformer Framework | Hugging Face Transformers |
| Audio Processing | Librosa |
| Data Analysis | NumPy, Pandas |
| Evaluation | Scikit-learn |
| Visualization | Matplotlib, Seaborn, Plotly |
| Web Application | Streamlit |
| Report Generation | ReportLab |
| Experimentation | Jupyter Notebook |

---

## Key Insight

The experiment investigates whether masking learned speech representations can improve the robustness of a speech command classifier in a low-resource setting.

The augmented configuration achieved higher test accuracy than the baseline in the evaluated experiment:

```text
Baseline Model       92%
        |
        | +6 percentage points
        v
Augmented Model      98%
```

This result suggests that latent-space masking can be a promising augmentation strategy for small-scale speech classification and motivates further evaluation on larger and more diverse datasets.

---

## Limitations

Several limitations should be considered when interpreting the results:

- The experiment focuses on only 10 speech command classes.
- The dataset is relatively small.
- The test set is limited in size.
- The recordings consist primarily of short speech commands.
- Only one latent-space masking strategy is evaluated.
- The experiment does not establish that latent-space masking is universally superior to conventional SpecAugment.
- Further evaluation is required on larger and more diverse speech datasets.

---

## Future Work

Potential extensions include:

- Evaluation on larger speech command datasets.
- Direct comparison with traditional spectrogram-based SpecAugment.
- Experiments with different masking ratios.
- Experiments with different masking durations.
- Robustness testing under background noise.
- Cross-speaker evaluation.
- Evaluation across multiple random seeds.
- Comparison of different pooling strategies.
- Further fine-tuning of the Wav2Vec2 backbone.
- Evaluation on more diverse real-world speech.

---

## Contributions

This project was developed collaboratively by a six-member team.

| Team Member | Contribution |
|---|---|
| Malak Tarek | Dataset preparation, preprocessing, and exploratory analysis |
| Shahd Abdelhay | Feature Engineering + wav2vec Input |
| Jana Ahmed | Wav2Vec2 model implementation and baseline development |
| Jomana El-Saghier | Time Mask Augmentation - inspired by SpecAugment |
| Adam Fadel | Training Pipeline + Experiment Runner |
| Maryam Ashraf | Documentation, experimentation, and project integration |

> Replace **Member 1–6** with the actual names and exact responsibilities of the six team members.

---

## Conclusion

This project presents an end-to-end speech command classification pipeline based on Wav2Vec2 and investigates the effect of latent-space feature masking in a low-resource setting.

The work combines:

- Pretrained speech representation learning
- Speech command classification
- Latent-space augmentation
- Baseline comparison
- Quantitative evaluation
- Per-class analysis
- Confusion matrix analysis
- Interactive Streamlit deployment
- Automated PDF reporting

The experimental results show that the augmented model achieved **98% test accuracy compared with 92% for the baseline** in the evaluated setting.

The project provides a foundation for further investigation into latent-space augmentation techniques for robust speech recognition.

---

## Acknowledgment

This project was developed as a collaborative machine learning and speech processing project.

---

## License

This repository is intended for educational and research purposes.

---

**Built with PyTorch, Wav2Vec2, Hugging Face Transformers, and Streamlit.**

</div>
