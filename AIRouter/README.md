# AIRouter

**AIRouter** is a Python-based project that leverages BERT for intelligent routing, classification, and performance benchmarking. It includes Jupyter notebooks and Python scripts to train models, classify text complexity, and measure inference latency—ideal for building routing logic or fallback systems using NLP.

---

## 📁 Repository Structure

- `training_data/`: Sample datasets used for training and testing.
- `bert_training_base_w_clincoos.ipynb`: Fine-tunes BERT on the CLINC150 dataset for base classification tasks.
- `bert_training_complexity_classification.ipynb`: Trains a BERT-based model to classify query complexity (e.g., simple vs complex).
- `complexity-classifier_wscores.py`: Uses the trained model to score inputs by complexity.
- `measure_latency.py`: Measures model inference latency on CPU.
- `putting_it_all_together.py`: Example workflow integrating all key components.

---

## ⚙️ Installation Instructions

1. **Clone the Repository**:
   ```bash
   git clone https://github.com/azure-bk-solutions/python-projects.git
   cd python-projects/AIRouter
   ```

2. **(Optional) Create and Activate a Virtual Environment**:
   ```bash
   python -m venv venv
   source venv/bin/activate       # Windows: venv\Scripts\activate
   ```

3. **Install Dependencies**:
   ```bash
   pip install -r requirements.txt
   ```

---

## 🚀 Usage

### ▶️ Run Notebooks Locally

1. Launch Jupyter Lab or Notebook:
   ```bash
   jupyter lab  # or jupyter notebook
   ```

2. Open one of the following notebooks:
   - `bert_training_base_w_clincoos.ipynb`
   - `bert_training_complexity_classification.ipynb`

3. Run cells sequentially. Make sure `training_data/` is populated.

---

### 🌐 Open in Google Colab

Click to open in Google Colab (ensure dataset is uploaded or accessible via Google Drive):

- **[Train Base BERT (CLINC150)](https://colab.research.google.com/github/azure-bk-solutions/python-projects/blob/main/AIRouter/bert_training_base_w_clincoos.ipynb)**
- **[Train Complexity Classifier](https://colab.research.google.com/github/azure-bk-solutions/python-projects/blob/main/AIRouter/bert_training_complexity_classification.ipynb)**

---

### 🧪 Run Python Scripts

After training:

```bash
# Score complexity of inputs
python complexity-classifier_wscores.py

# Measure inference latency
python measure_latency.py

# Full workflow pipeline
python putting_it_all_together.py
```

---

## 📌 Notes

- CPU-only inference is supported but may be slow for large batches.
- Datasets are expected in `training_data/`—adjust paths if needed.
- The model checkpoints and tokenizer may download from HuggingFace on first run.

---

## 📜 License

MIT License. See `LICENSE` file for more details.

---

## 🤝 Contributions

Feel free to submit issues or PRs! For major changes, open a discussion first.

---

## 👤 Maintainer

Built by [azure-bk-solutions](https://github.com/azure-bk-solutions)
