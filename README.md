# ML 101 · Class 1: Lines That Learn

A 90-minute introduction to linear regression, logistic regression and the perceptron. All three are trained on the Iris dataset as a binary classification problem: *setosa* vs *versicolor*, using sepal length and sepal width.

## Contents

| File | What it is |
|---|---|
| `lecture.html` | Lecture notes with a timing plan, the formulas, and live in-browser training demos. Open it in a browser. |
| `ml101_class1.ipynb` | Class notebook. Each model is written in plain NumPy, then trained and animated inline. |
| `ml101/common.py` | Data loading, standardization and the animation player used by the notebook. |
| `data/iris.csv` | Fisher's Iris dataset (150 flowers, measurements in cm). |
| `figures/` | Pre-rendered GIFs of each animation, plus a comparison plot, for slides. |

## Running the notebook

```bash
pip install -r requirements.txt
jupyter notebook ml101_class1.ipynb   # or open it in VS Code
```

Run the cells from top to bottom (`Shift + Enter`). Each model ends with an animation cell: press ▶ under the plot to watch the decision boundary converge.

To try other problems, edit `CLASSES`, `FEATURES` or `LR` in the notebook. For example, `CLASSES = ("versicolor", "virginica")` gives overlapping species, and the perceptron never converges on them.

## Results (setosa vs versicolor, sepal features)

| Model | Training accuracy |
|---|---|
| Linear regression (threshold at 0.5) | 99% |
| Logistic regression | 100% |
| Perceptron | 100% (13 updates) |

The notation follows the CS229 lecture notes by Tengyu Ma and Andrew Ng (chapters 1 and 2).
