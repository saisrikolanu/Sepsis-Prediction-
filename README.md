Interpretable AI for Early Sepsis Prediction using Counterfactual Explanations
A multimodal deep learning framework for early sepsis prediction in ICU settings, combining structured physiological time-series with ClinicalBERT-encoded clinical notes, with counterfactual explainability via DiCE.
Author: Sai Sri Kolanu — University at Buffalo (saisriko@buffalo.edu)

Overview
Early sepsis detection is a critical challenge in intensive care — subtle symptom onset, delayed lab confirmation, and fragmented EHR documentation make timely prediction difficult. This project presents a comprehensive evaluation of classical ML, deep learning, and multimodal fusion models for sepsis prediction, concluding with counterfactual explainability analysis to make model decisions transparent and clinically interpretable.
Best result: Multimodal LSTM (structured time-series + ClinicalBERT) achieves F1 = 0.8056 and AUPRC = 0.8290 on MIMIC-IV, outperforming all unimodal and classical baselines.

Datasets
PhysioNet 2019 Challenge

40 ICU variables sampled hourly
24 most informative features selected (labs + vitals)
Sepsis labels constructed per Sepsis-3 definition

MIMIC-IV Structured Data

48-hour time windows
24 features × 3 channels (value, mask, delta) → shape: (N, 48, 24)

MIMIC-IV Clinical Notes

Embedded using ClinicalBERT (768-dimensional vectors)
Mean-pooled per ICU stay for multimodal fusion
Train: 3,992,363 embeddings | Val: 856,667 | Test: 856,779


Models
Classical Machine Learning (PhysioNet)
ModelAUROCAUPRCAccuracyF1Logistic Regression0.76640.82280.70040.7534Random Forest0.86850.86760.78490.7774XGBoost0.76260.81880.69770.7521
Deep Learning on Structured Time Series (PhysioNet)
Trained on 48×24 structured sequences:
ModelHighlightLSTMBest AUROC = 0.8868, AUPRC = 0.8871Bi-LSTMBest F1 = 0.7928, Recall = 0.8788Transformer / TCNAUROC ≈ 0.875, AUPRC ≈ 0.874CNNAUPRC ≈ 0.078 — failed to learn sepsis patternsGRU, Compact LSTMCompetitive but below LSTM/Bi-LSTM
MIMIC-IV Unimodal Structured Models
ModelAUROCAUPRCF1Logistic Regression0.7180.8100.701Compact LSTM0.6930.7950.682Transformer0.74250.82570.8033
Multimodal LSTM — Structured + ClinicalBERT (MIMIC-IV)
Fusion: Z = ReLU(W1·h_t + W2·e_bert) → ŷ = σ(W3·z)
Classification threshold = 0.30 (tuned for best validation F1).
SplitAUROCAUPRCAccuracyF1Train0.75160.83060.69940.8019Val0.74750.83020.70440.8067Test0.74710.82900.70380.8056

Key finding: Multimodal LSTM achieves the highest F1 among all MIMIC-IV models. Clinical notes clearly improve detection sensitivity and precision.


Counterfactual Explainability (DiCE)
To understand model decisions, we use DiCE to generate diverse counterfactuals — the minimal perturbation to an input that flips the model's prediction from non-septic (0) to septic (1).
Example: Query Instance (Prediction = 0 → Non-Septic)
FeatureOriginalCounterfactualHR (normalized)−0.281966−0.261796WBC (normalized)−0.149762−0.149762 (unchanged)
Only a small upward shift in heart rate was sufficient to flip the label — WBC was not needed.
Temporal Visualization Findings
Heart Rate (HR): Counterfactual HR rises earlier and more sharply than the original. Even small HR increases lead to high-risk classification.
White Blood Cell Count (WBC): Original WBC drops sharply after hour 11; counterfactual WBC stays flat. The model interprets a stable WBC as a higher-risk signal.
Interpretations

Decision Boundary Proximity — All five counterfactuals are nearly identical, indicating the patient lies very close to the decision threshold.
Multimodal Influence — The small change needed in structured features implies ClinicalBERT embeddings contribute heavily to the risk prediction.
High Sensitivity — The model responds strongly to minor physiological changes, reinforcing the importance of explainability for safe clinical deployment.


Project Structure
├── data/
│   ├── physionet/              # PhysioNet 2019 Challenge data
│   └── mimic_iv/               # MIMIC-IV structured + clinical notes
├── embeddings/
│   └── clinicalbert_embed.py   # ClinicalBERT embedding pipeline
├── models/
│   ├── classical.py            # Logistic Regression, Random Forest, XGBoost
│   ├── temporal.py             # GRU, LSTM, Bi-LSTM, CNN, TCN, Transformer
│   └── multimodal_lstm.py      # Multimodal fusion architecture
├── explainability/
│   └── counterfactual_dice.py  # DiCE counterfactual generation & visualization
├── train.py                    # Training entry point
├── evaluate.py                 # Evaluation script (AUROC, AUPRC, F1)
└── README.md

Getting Started
Prerequisites
bashpip install torch transformers scikit-learn xgboost dice-ml pandas numpy matplotlib
Train a Model
bash# Classical models on PhysioNet
python train.py --dataset physionet --model random_forest

# Multimodal LSTM on MIMIC-IV
python train.py --dataset mimic_iv --model multimodal_lstm --threshold 0.30
Generate Counterfactuals
bashpython explainability/counterfactual_dice.py \
  --model checkpoints/multimodal_lstm.pt \
  --query data/mimic_iv/sample_query.csv \
  --n_counterfactuals 5

Key Takeaways

Deep sequential models (LSTM, Bi-LSTM) significantly outperform classical ML on structured physiological data.
Multimodal fusion with ClinicalBERT embeddings improves F1 and AUPRC beyond the best unimodal models on MIMIC-IV.
Counterfactual explainability reveals that the model is highly sensitive near decision boundaries, with ClinicalBERT context driving much of the risk signal.
Interpretable AI is essential for safe clinical deployment — minor physiological changes can alter predictions, and clinicians need to understand why.


References

Sepsis-3 Definition: Singer et al., JAMA, 2016.
ClinicalBERT: Huang et al., 2019. https://arxiv.org/abs/1904.05342
DiCE (Diverse Counterfactual Explanations): Mothilal et al., 2020. https://arxiv.org/abs/1905.07697
PhysioNet 2019 Challenge: https://physionet.org/content/challenge-2019/
MIMIC-IV: Johnson et al., 2023. https://physionet.org/content/mimiciv/
