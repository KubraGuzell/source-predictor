# 🚀 Source Prediction AI (DistilBERT + ONNX + Streamlit)

This project is an end-to-end, production-ready Machine Learning system designed to predict the source or author (e.g., Obama, Trump, Hemingway, etc.) of a given text. 

The system features a **DistilBERT** model fine-tuned for sequence classification, optimized via **ONNX** for high-speed inference and low resource consumption, and deployed as an interactive web application using **Streamlit**.

---

### 🌐 Live Deployment

The system is fully containerized (Docker) and hosted on Hugging Face Spaces.

* **Interactive Web App:** Test the model in real-time here:  
  👉 **[https://kubraguzel-source-predictor-api.hf.space](https://kubraguzel-source-predictor-api.hf.space)**

---

### 🧠 Technical Approaches & Methodology

#### 1. Solving Data Leakage: Mathematical Clustering
* **The Challenge:** In the raw dataset, consecutive paragraphs from the same original document (e.g., a single speech) created a high risk of **Data Leakage**. A random `train_test_split` would allow the model to "memorize" specific documents rather than learning general authorial styles.
* **The Solution:** To ensure robust generalization, a **Clustering** method based on thematic similarities (TF-IDF Cosine Similarity) was developed. Similar texts were grouped under a `cluster_id`. The dataset was then split using Scikit-learn's `GroupShuffleSplit`, ensuring that entire clusters (documents) remained together in either the training or validation set. This is the core engineering decision of the project.

#### 2. Model Optimization: DistilBERT + ONNX Runtime
* **Efficiency:** **DistilBERT** was selected for being 40% smaller and 60% faster than standard BERT while retaining 97% of its performance.
* **Production Readiness:** To eliminate the heavy overhead of PyTorch in production, the model was exported to **ONNX** format. This allows for millisecond-level inference on low-cost CPU instances using the **ONNX Runtime** engine.

#### 3. Hyperparameter Tuning (Ablation Study)
| Experiment | Weight Decay | Warmup Ratio | Metric Choice | Outcome |
| :--- | :---: | :---: | :---: | :--- |
| **Exp 1** | 0.01 | None | `macro_f1` | Solved leakage, but showed signs of early overfitting. |
| **Exp 2** | 0.05 | 0.1 | `macro_f1` | High regularization led to slight underfitting. |
| **Exp 3 (Champion)** | **0.01** | **None** | **`eval_loss`** | **Optimal Balance:** Best generalization achieved by monitoring `eval_loss`. |

---

### 📊 Model Performance Analysis (Validation Set)

The following metrics represent the final evaluation of the **ONNX-optimized DistilBERT** model on the held-out validation set.

#### 📈 Global Metrics
| Metric | Value |
| :--- | :---: |
| **Overall Accuracy** | **90.32%** |
| **Macro F1-Score** | **89.18%** |
| **Weighted F1-Score** | **90.11%** |

#### 📝 Detailed Classification Report
| Class | Precision | Recall | F1-Score | Support |
| :--- | :---: | :---: | :---: | :---: |
| **blog11518** | 0.98 | 0.97 | 0.98 | 588 |
| **blog25872** | 0.95 | 0.86 | 0.91 | 72 |
| **blog30102** | 0.89 | 0.72 | 0.80 | 112 |
| **blog30407** | 0.90 | 0.92 | 0.91 | 210 |
| **blog5546** | 0.75 | 0.91 | 0.82 | 175 |
| **Bush** | 0.66 | 0.84 | 0.74 | 148 |
| **Fitzgerald** | 0.91 | 0.94 | 0.92 | 681 |
| **h** | 1.00 | 0.91 | 0.95 | 22 |
| **Hemingway** | 0.94 | 0.91 | 0.93 | 396 |
| **Obama** | 0.89 | 0.76 | 0.82 | 250 |
| **pp** | 0.94 | 0.88 | 0.91 | 33 |
| **qq** | 1.00 | 1.00 | 1.00 | 16 |
| **Trump** | 0.87 | 0.86 | 0.87 | 327 |
| **Woolf** | 0.95 | 0.93 | 0.94 | 378 |

---

### 🧠 Performance Insights & Error Analysis

* **Linguistic Distinctness:** The model achieves exceptional performance on authors with unique syntactic structures like **Hemingway** (short, declarative sentences) and **Woolf** (stream of consciousness), with F1-scores of **0.93** and **0.94** respectively.
* **Political Domain Challenges:** The lower precision for **Bush (0.66)** compared to his recall **(0.84)** indicates a model bias where generic political rhetoric is sometimes misclassified as Bush. This highlights the linguistic proximity and shared "Presidential" vocabulary between figures like Obama and Bush.
* **Clustering Success:** High accuracy across various **blog** categories confirms that the **Clustering-based Data Splitting** effectively prevented data leakage. The model learned stylistic features rather than memorizing document-specific patterns.

---

### 🏗️ Project Structure

```text
/
├── app/
│   └── main.py              # Streamlit Web UI and ONNX Runtime implementation
├── data_analysis/           # Exploratory Data Analysis (EDA)
├── src/
│   ├── data_split.py       # Clustering-based data partitioning script
│   ├── export_onnx.py      # PyTorch to ONNX conversion utility
│   ├── train.py            # Model fine-tuning script (Hugging Face Trainer)
│   └── data_utils.py       # Preprocessing and utility functions
├── requirements.txt         # Production dependencies
├── Dockerfile               # Deployment configuration for Hugging Face Spaces
└── README.md                # Project documentation