import numpy as np
from sklearn.datasets import make_moons, make_circles, make_classification

def generate_moons(n_samples=300, noise=0.2, random_state=42):
    X, y = make_moons(n_samples=n_samples, noise=noise, random_state=random_state)
    return X, y

def generate_circles(n_samples=300, noise=0.1, factor=0.5, random_state=42):
    X, y = make_circles(n_samples=n_samples, noise=noise, factor=factor, random_state=random_state)
    return X, y

def generate_linear(n_samples=300, noise=1.0, random_state=42):
    X, y = make_classification(
        n_samples=n_samples,
        n_features=2,
        n_redundant=0,
        n_informative=2,
        n_clusters_per_class=1,
        flip_y=0.01,
        class_sep=1.5,
        random_state=random_state
    )
    return X, y
