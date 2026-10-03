# ML Model Testing with Python + pytest

[![tests](https://github.com/vaishnavi-kalmegh/ml-model-testing-pytest/actions/workflows/tests.yml/badge.svg)](https://github.com/vaishnavi-kalmegh/ml-model-testing-pytest/actions/workflows/tests.yml)

A QA-focused project: train a small customer-churn classifier with scikit-learn, then try to break its prediction interface with pytest. The point is not the model's accuracy. It is proving the pipeline fails loudly on bad data and stays stable on unfamiliar-but-valid data.

## What gets tested

**29 automated test cases collected by pytest**

| Area | Test cases | Coverage |
|---|---:|---|
| Happy path | 5 | Valid predictions, binary labels, row-level shape, single valid record |
| Bad inputs | 3 | Empty/non-DataFrame input and unexpected columns |
| Missing data | 4 | Missing columns, all numeric values missing, partial NaNs, `None` vs `NaN` |
| Wrong data types | 2 | Text supplied to numeric features |
| Unseen categories | 3 | New plan/region values plus case/whitespace variants |
| Quality and behaviour | 11 | Column order, mutation safety, row/batch consistency, reproducibility, label type, quality floor, numeric extremes |
| Persistence | 1 | Save and reload the trained model |

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

GitHub Actions runs on every push and pull request: install dependencies, run the self-contained pytest suite, and upload the HTML report as a workflow artifact. The tests train their own session-scoped model, so CI does not depend on a previously generated model file.

## Sample run

<!-- Add a real screenshot after running pytest, then uncomment: -->
<!-- ![pytest results](docs/pytest-results.png) -->

## Author

**Vaishnavi Kalmegh**
