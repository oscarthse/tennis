import pytest
import numpy as np
from sklearn.linear_model import LogisticRegression
from sklearn.datasets import make_classification

def test_regularization_shrinks_weights():
    """
    Verify that stronger regularization (lower C) results in smaller weight norms.
    """
    # 1. Setup Data
    X, y = make_classification(n_samples=100, n_features=20, random_state=42)

    # 2. Train Weak Regularization Model (High C)
    weak_reg_model = LogisticRegression(C=100.0)
    weak_reg_model.fit(X, y)
    weak_norm = np.linalg.norm(weak_reg_model.coef_)

    # 3. Train Strong Regularization Model (Low C)
    strong_reg_model = LogisticRegression(C=0.01)
    strong_reg_model.fit(X, y)
    strong_norm = np.linalg.norm(strong_reg_model.coef_)

    # 4. Assert
    # The weights of the strong model should be significantly smaller
    # We expect at least a 50% reduction in the norm for this synthetic dataset
    reduction_ratio = strong_norm / weak_norm
    assert reduction_ratio < 0.5, f"Weights only reduced by {1-reduction_ratio:.1%}, expected >50%"
    print(f"Weak Norm: {weak_norm:.2f}, Strong Norm: {strong_norm:.2f} (Reduction: {1-reduction_ratio:.1%})")

def test_l1_sparsity():
    """
    Verify that L1 regularization produces more zero weights than L2.
    """
    X, y = make_classification(n_samples=100, n_features=50, random_state=42)

    # L1 Model
    l1_model = LogisticRegression(penalty='l1', C=0.1, solver='liblinear')
    l1_model.fit(X, y)
    l1_zeros = np.sum(l1_model.coef_ == 0)

    # L2 Model
    l2_model = LogisticRegression(penalty='l2', C=0.1, solver='lbfgs')
    l2_model.fit(X, y)
    l2_zeros = np.sum(l2_model.coef_ == 0)

    assert l1_zeros > l2_zeros, "L1 did not produce more sparsity than L2!"
