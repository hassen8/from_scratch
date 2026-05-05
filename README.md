# ML From Scratch

Implementations of machine learning algorithms using only NumPy and SciPy — no sklearn, no torch, no shortcuts.

## Structure

```
scratch/
├── supervised/
│   ├── linear/         # Linear Regression, Ridge, Lasso, Logistic Regression
│   ├── trees/          # Decision Tree, Random Forest, Gradient Boosting
│   └── neural/         # MLP
└── unsupervised/
    ├── clustering/     # K-Means, DBSCAN
    └── decomposition/  # PCA, Truncated SVD
```

## Algorithms

### Supervised
| Algorithm | Module | Status |
|---|---|---|
| Linear Regression | `supervised/linear/linear_regression.py` | 🔲 |
| Ridge Regression | `supervised/linear/ridge.py` | 🔲 |
| Lasso | `supervised/linear/lasso.py` | 🔲 |
| Logistic Regression | `supervised/linear/logistic_regression.py` | 🔲 |
| Decision Tree | `supervised/trees/decision_tree.py` | 🔲 |
| Random Forest | `supervised/trees/random_forest.py` | 🔲 |
| Gradient Boosting | `supervised/trees/gradient_boosting.py` | 🔲 |
| MLP | `supervised/neural/mlp.py` | 🔲 |

### Unsupervised
| Algorithm | Module | Status |
|---|---|---|
| K-Means | `unsupervised/clustering/kmeans.py` | 🔲 |
| DBSCAN | `unsupervised/clustering/dbscan.py` | 🔲 |
| PCA | `unsupervised/decomposition/pca.py` | 🔲 |
| Truncated SVD | `unsupervised/decomposition/svd.py` | 🔲 |

## Usage

```python
from scratch.supervised.linear.linear_regression import LinearRegression

model = LinearRegression()
model.fit(X_train, y_train)
preds = model.predict(X_test)
```

All estimators follow a consistent interface:
- Supervised: `fit(X, y)` → `predict(X)`
- Unsupervised: `fit(X)` → `transform(X)` or `fit_predict(X)`

## Setup

```bash
pip install -r requirements.txt
pytest tests/
```

## Dependencies

- `numpy` — core numerics
- `scipy` — sparse ops, special functions
- `jupyter` — notebooks
- `pytest` — tests
