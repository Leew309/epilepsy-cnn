# epilepsy-cnn
# Epilepsy CNN (1D-CNN) — Seizure vs Non-Seizure Classification

A 1D Convolutional Neural Network (PyTorch) for binary classification of epileptic seizure events using tabular EEG-derived features.

## Dataset
- Input: CSV with **178 features** and a target column `y`.
- Original labels: `[1,2,3,4,5]`
- Binary conversion:
  - `y = 1` → **Seizure**
  - `y != 1` → **Non-Seizure**
- Class distribution after conversion:
  - Non-Seizure: 9200
  - Seizure: 2300

## Model (ImprovedCNN1D)
Architecture:
- Conv1D(1→32) + BatchNorm + ReLU + MaxPool + Dropout
- Conv1D(32→64) + BatchNorm + ReLU + MaxPool + Dropout
- Conv1D(64→128) + BatchNorm + ReLU + MaxPool + Dropout
- Global Average Pooling
- FC(128→64) + Dropout
- FC(64→32) + Dropout
- FC(32→1) + Sigmoid

Regularization:
- Dropout at input + after conv blocks + strong dropout in FC layers
- AdamW optimizer + weight decay
- ReduceLROnPlateau scheduler
- Early stopping

## Training
- Stratified K-Fold Cross Validation: **2 folds**
- Feature scaling: StandardScaler fitted on train split only
- Imbalance handling:
  - Class weights computed from dataset distribution
  - WeightedRandomSampler used in DataLoader

## Evaluation
Metrics reported across folds:
- AUC: **0.9941 ± 0.0009**
- Recall: **0.9142 ± 0.0041**
Also includes:
- Training vs Validation loss curves
- Confusion matrix per fold

## How to Run (Colab)
1. Upload the dataset file to Colab (or place it at):
   `/content/Epileptic Seizure Recognition.csv`
2. Run all cells.

## Notes / Possible Improvements
- Keep validation/test loaders unweighted (use WeightedRandomSampler only for training).
- Use one consistent decision threshold across validation and evaluation.
- Save the best model based on validation recall (or F1/AUC) consistently.
- Add Precision/F1 and PR-AUC reporting due to class imbalance.
