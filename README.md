# ML Model Testing with Python + pytest

[![tests](https://github.com/vaishnavi-kalmegh/ml-model-testing-pytest/actions/workflows/tests.yml/badge.svg)](https://github.com/vaishnavi-kalmegh/ml-model-testing-pytest/actions/workflows/tests.yml)

A QA-focused project: train a small customer-churn classifier with scikit-learn, then try to break its prediction interface with pytest. The point is not the model's accuracy. It is proving the pipeline fails loudly on bad data and stays stable on unfamiliar-but-valid data.

## What gets tested

| Area | Scenarios |
|---|---|
| Happy path | Valid predictions, correct shape and labels |
| Bad inputs | Empty input, non-DataFrame input, unexpected columns |
| Missing data | Missing columns, partial NaNs, all numeric values missing |
| Wrong data types | Text supplied to numeric features |
| Unseen categories | New plan / region values at inference time |
| Model contracts | Reproducibility, integer/binary outputs |
| Persistence | Save and reload the trained model |

## Key design decision: invalid vs. unfamiliar

The suite separates **invalid data** from **valid but unfamiliar data**.

- Text in a numeric column breaks the data contract, so it is rejected with a clear `TypeError`.
- An unseen category such as `enterprise` is valid input the model has simply never seen, so it is accepted (`OneHotEncoder(handle_unknown="ignore")`).
- Partial missing numeric values are filled by median imputation.
- A row where *every* numeric feature is missing has no usable signal, so it is rejected.

## Project structure

```
model/classifier.py          # pipeline + train/predict/save/load
tests/test_classifier.py     # pytest suite
train.py                     # trains and saves the model
pytest.ini                   # pytest config, markers, HTML report
.github/workflows/tests.yml  # CI
```

## Run locally

```bash
python -m venv .venv

# Windows (PowerShell)
.venv\Scripts\Activate.ps1
# macOS / Linux
source .venv/bin/activate

pip install -r requirements.txt
python train.py
pytest
```

The HTML report is written to `reports/report.html`.

Run a single category with markers, e.g. `pytest -m unseen_category`.

## CI

GitHub Actions runs on every push and pull request: install dependencies, train the model, run the full suite, and upload the HTML report as a workflow artifact.

## Sample run

<!-- Add a real screenshot after running pytest, then uncomment: -->
<!-- ![pytest results](docs/pytest-results.png) -->

## Author

**Vaishnavi Kalmegh**
