"""This script is for the introduction of cutting-edge foundational
tabular models that leverage pre-training and in-context learning to
achieve state-of-the-art performance on tabular datasets. These models
represent a significant advancement in automated ML for structured data.

First, install AutoGluon with support for foundational models.
!pip install uv
!uv pip install autogluon.tabular[mitra]
!uv pip install autogluon.tabular[tabicl]
!uv pip install autogluon.tabular[tabpfn]
"""

import pandas as pd
from autogluon.tabular import TabularDataset, TabularPredictor
from sklearn.model_selection import train_test_split
from sklearn.datasets import load_wine, fetch_california_housing

# Example data: Wine Dataset (multi-class classification) and California
# Housing (regression).

# Load datasets

# 1. Wine (Multi-class Classification)
wine_data = load_wine()
wine_df = pd.DataFrame(wine_data.data, columns=wine_data.feature_names)
wine_df['target'] = wine_data.target

# 2. California Housing (Regression)
housing_data = fetch_california_housing()
housing_df = pd.DataFrame(housing_data.data, columns=housing_data.feature_names)
housing_df['target'] = housing_data.target

print("Dataset shapes:")
print(f"Wine: {wine_df.shape}")
print(f"California Housing: {housing_df.shape}")

# Create train/test splits (80/20)
wine_train, wine_test = train_test_split(
    wine_df,
    test_size=0.2,
    random_state=42,
    stratify=wine_df['target']
)
housing_train, housing_test = train_test_split(housing_df, test_size=0.2, random_state=42)

print("Training set sizes:")
print(f"Wine: {len(wine_train)} samples")
print(f"Housing: {len(housing_train)} samples")

# Convert to TabularDataset
wine_train_data = TabularDataset(wine_train)
wine_test_data = TabularDataset(wine_test)
housing_train_data = TabularDataset(housing_train)
housing_test_data = TabularDataset(housing_test)

# Mitra: AutoGluon's Tabular Foundation Model.
# Mitra is a new SOTA tabular foundation model developed by the AutoGluon team,
# natively supported in AutoGluon with just 3 lines of code via `predictor.fit()`.
# Built on the in-context learning paradigm and pretrained exclusively on synthetic data,
# Mitra introduces a principled pretraining approach by carefully selecting and mixing
# diverse synthetic priors to promote robust generalization across a wide range of
# real-world tabular datasets.

# In particular, it excels on small tabular datasets with fewer than 5000 samples and
# 100 features, for both classification and regression tasks. It supports both zero-shot
# fine-tuning modes and runs seamlessly on both GPU and CPU. Its weights are fully open-sourced
# under the Apache-2.0 licence, making it a privacy-conscious and production-ready solution
# for enterprises concerned about data sharing and hosting.

# Using Mitra for classification.
# Create predictor with Mitra
print("Training Mitra classifier on classification dataset...")
mitra_predictor = TabularPredictor(label='target')
mitra_predictor.fit(
    wine_train_data,
    hyperparameters={
        'MITRA': {'fine_tune': False}
    },
)

print("\nMitra training completed!")

# Evaluate Mitra performance.
# Make predictions
mitra_predictions = mitra_predictor.predict(wine_test_data)
print("Sample Mitra predictions:")
print(mitra_predictions.head(10))

# Show prediction probabilities for first few samples
mitra_predictions = mitra_predictor.predict_proba(wine_test_data)
print(mitra_predictions.head())

# Show model leaderboard
print("\nMitra Model Leaderboard:")
print(mitra_predictor.leaderboard(wine_test_data))

# Fine-tuning with Mitra.
mitra_predictor_ft = TabularPredictor(label='target')
mitra_predictor_ft.fit(
    wine_train_data,
    hyperparameters={
        'MITRA': {'fine_tune': True, 'fine_tune_steps': 10}
    },
    time_limit=120,  # 2 minutes
   )

print("\nMitra fine-tuning completed!")

# Evaluating Fine-tuned Mitra performance.
# Show model leaderboard
print("\nMitra Model Leaderboard:")
print(mitra_predictor_ft.leaderboard(wine_test_data))
