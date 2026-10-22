# StudentScore AI

A machine-learning project that estimates student exam scores from academic and background factors.

## Project contents

- `app.py` — Streamlit prediction interface.
- `notebooks/StudentScore_AI.ipynb` — model exploration, training and evaluation.
- `models/studentscore_ridge_model.joblib` — serialized trained model.
- `data/StudentPerformanceFactors.csv` — dataset used by the project.
- `requirements.txt` — Python dependencies.

## Model

The notebook trains a Ridge Regression pipeline with preprocessing and `alpha=10.0`.

Reported evaluation on the original test split:
- MAE: approximately 0.417
- RMSE: approximately 1.521
- R²: approximately 0.825

These metrics describe the evaluated test split; R² is not classification accuracy and does not guarantee performance on new data.

## Run locally

Use Python 3.11 or 3.12 for broad compatibility with the ML dependencies.

```bash
python -m venv .venv
```

Windows PowerShell:

```powershell
.\.venv\Scripts\Activate.ps1
pip install -r requirements.txt
streamlit run app.py
```

## Deploy on Streamlit Community Cloud

1. Push this repository to GitHub.
2. Create a new app in Streamlit Community Cloud.
3. Select the repository and branch.
4. Set the main file path to `app.py`.
5. Deploy.

Ensure the model file is included in the repository and the dataset can legally be redistributed before making the repository public.

## Limitations

Predictions are estimates and may be inaccurate for individual students. The model should not be used as the sole basis for educational decisions.
