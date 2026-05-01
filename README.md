# 🎙️ Speech Analysis for Depression Assessment

Deep-learning and classical baselines for depression prediction from speech, developed for the Intelligent Signal Analysis course (University of Palermo, LM-32).

## 👥 Authors

- Andrea Spinelli
- Antonio Spedito
- Davide Bonura

## 🧠 Approach

The project compares multiple paradigms:

1. **SVM baseline** on handcrafted acoustic features
2. **1D-CNN** on raw waveform chunks
3. **SSL-based hierarchical model** (wav2vec2 / HuBERT / WavLM-like representations + sequential modeling + gated attention pooling)

## 📁 Structure

```text
.
├── config.yaml
├── requirements.txt
├── scripts/
│   ├── preprocessor.py
│   ├── feature_extraction.py
│   ├── train.py
│   ├── test.py
│   └── plot_results.py
└── src/
    ├── trainer.py
    ├── evaluator.py
    ├── preprocessor.py
    └── ...
```

## 🚀 Setup

```bash
pip install -r requirements.txt
```

## ▶️ Run

1. Configure the active experiment in `config.yaml` (`svm`, `cnn`, or `ssl`)
2. Preprocess data:

```bash
python -m scripts.preprocessor
```

3. (Optional for SSL) extract features:

```bash
python -m scripts.feature_extraction
```

4. Train and test:

```bash
python -m scripts.train
python -m scripts.test
```

5. Plot aggregate results:

```bash
python -m scripts.plot_results
```

## 📌 Notes

- The workflow is configuration-driven and supports K-fold runs
- Hyperparameter search and layer sweeps are available via config flags
- Intermediate features, saved models, and results are generated in dedicated folders

