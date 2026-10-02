# ML Model Testing with Python + pytest

A portfolio project demonstrating QA testing of a machine-learning classification pipeline, including data-contract validation and edge cases.

## Project goal

This project trains a small binary customer-churn classifier with scikit-learn and tests its prediction interface with pytest. The preprocessing pipeline handles numeric missing values and categorical values, including categories that were not present during training.

## QA coverage

**18 automated tests**

| Area | Coverage |
|---|---|
| Happy path | Valid predictions, shape and labels |
| Bad inputs | Empty input, non-DataFrame input, unexpected columns |
| Missing data | Missing columns, partial missing values, all numeric values missing |
| Wrong data types | Text supplied to numeric features |
| Unseen categories | New plan/region values |
| Model quality contracts | Reproducibility and integer/binary outputs |
| Persistence | Save and reload trained model |

## Technology

- Python
- pandas
- NumPy
- scikit-learn
- joblib
- pytest
- pytest-html
- GitHub Actions

## Project structure

    model/classifier.py
    tests/test_classifier.py
    train.py
    requirements.txt
    pytest.ini
    .github/workflows/tests.yml

## Run locally

    python -m venv .venv
    .venv\Scripts\Activate.ps1
    pip install -r requirements.txt
    python train.py
    pytest

The self-contained HTML report is generated at reports/report.html.

## ML-specific testing approach

The suite separates invalid data contracts from valid but unfamiliar data.

A numeric feature supplied as text is rejected with a clear TypeError. In contrast, an unseen category such as enterprise is accepted because OneHotEncoder uses handle_unknown="ignore". This verifies that the inference pipeline is robust to categories that were not present in the training dataset.

Partial missing numeric values are supported through median imputation. An input where every numeric feature is missing is rejected because it contains no usable numeric signal.

## CI/CD

GitHub Actions runs on every push and pull request. The workflow:

1. Installs Python and dependencies.
2. Trains the classifier.
3. Runs the complete pytest suite.
4. Generates a self-contained HTML report.
5. Uploads the report as a workflow artifact.

## Portfolio evidence

After a real execution, useful evidence screenshots include:

- Repository structure
- Terminal pytest result
- pytest HTML report
- GitHub Actions successful run
- Example unseen-category test

Execution screenshots should be captured from a real run; fabricated evidence is intentionally not included.

## Author

**Vaishnavi Kalmegh**
