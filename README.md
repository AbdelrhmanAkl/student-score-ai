# StudentScore AI
### Machine Learning–Powered Student Exam Score Prediction

StudentScore AI is an interactive machine learning application that predicts a student's estimated exam score from academic performance, study habits, lifestyle indicators, and student background factors.

Built with Python, Scikit-learn, Pandas, Joblib, and Streamlit, the project demonstrates how a trained regression pipeline can be integrated into a user-friendly web application.

<p align="center">
  <a href="https://student-score-ai.streamlit.app/"><strong>Live Demo</strong></a> ·
  <a href="https://github.com/AbdelrhmanAkl/student-score-ai"><strong>Source Code</strong></a>
</p>

---

## Overview

The application accepts student-related information and passes the input through a saved Scikit-learn pipeline to produce an estimated exam score.

**Key capabilities**
- Predict exam scores through an interactive web interface.
- Accept numerical academic and lifestyle inputs.
- Collect categorical student background information.
- Load a serialized Ridge Regression pipeline with Joblib.
- Run as a web application using Streamlit Community Cloud.

## Technology Stack

| Technology | Purpose |
|---|---|
| Python | Application and model integration |
| Pandas | Tabular data processing |
| Scikit-learn | Preprocessing and machine learning |
| Ridge Regression | Score prediction |
| Joblib | Model serialization and loading |
| Streamlit | Interactive web interface |
| Git & GitHub | Version control and source hosting |
| Streamlit Community Cloud | Web deployment |

## Dataset

The project uses the **Student Performance Factors** dataset, containing 6,607 records and 20 columns, including the target variable `Exam_Score`.

The dataset covers factors such as:
- Study hours, attendance, and previous scores
- Tutoring sessions, sleep, and physical activity
- Motivation and extracurricular activities
- Access to educational resources and internet access
- Family income, parental education, and school-related factors

The application uses the dataset to populate categorical input options. The trained model receives the corresponding feature values to generate a prediction.

> Dataset note: Review the dataset's original license and redistribution terms before reusing it in other projects.

## How It Works

1. The user enters academic and background information.
2. The application organizes the values into a Pandas DataFrame.
3. The saved preprocessing and regression pipeline transforms the input features.
4. Ridge Regression generates the estimated exam score.
5. Streamlit displays the prediction and a disclaimer.

## Model

The application uses a trained **Ridge Regression** pipeline.

Ridge Regression is a regularized linear regression algorithm that applies an L2 penalty to help control coefficient magnitudes. The saved pipeline includes preprocessing components, so input preparation remains consistent with the training workflow.

Model performance should be evaluated on a held-out test set using appropriate regression metrics:

- **MAE:** Mean Absolute Error
- **RMSE:** Root Mean Squared Error
- **R²:** Coefficient of Determination

No performance figures are claimed here without confirming the final model artifact and its evaluation results.

## Repository Structure

```text
student-score-ai/
├── app.py
├── requirements.txt
├── README.md
├── .gitignore
├── data/
│   └── StudentPerformanceFactors.csv
├── models/
│   └── studentscore_ridge_model.joblib
└── notebooks/
    └── StudentScore_AI.ipynb
```

## Run Locally

### 1. Clone the repository

```bash
git clone https://github.com/AbdelrhmanAkl/student-score-ai.git
cd student-score-ai
```

### 2. Create a virtual environment

Python 3.12 is a recommended starting point for the pinned Scikit-learn dependency.

**Windows PowerShell**
```powershell
py -3.12 -m venv .venv
.\.venv\Scripts\Activate.ps1
```

**macOS / Linux**
```bash
python3.12 -m venv .venv
source .venv/bin/activate
```

### 3. Install dependencies

```bash
python -m pip install --upgrade pip
pip install -r requirements.txt
```

### 4. Launch the application

```bash
streamlit run app.py
```

Open the local URL printed in the terminal, typically `http://localhost:8501`.

Ensure the model and dataset files are available at the paths expected by the application.

## Deployment

The application is configured for deployment through Streamlit Community Cloud.

1. Push the project to GitHub.
2. Create an application in Streamlit Community Cloud.
3. Select this repository and the `main` branch.
4. Set the main file path to `app.py`.
5. Select a Python version compatible with the pinned dependencies.
6. Deploy and review the application logs if any errors occur.

The model was serialized with Scikit-learn 1.6.1, so the deployment environment should use a compatible version to avoid model-loading errors.

## Limitations and Responsible Use

- Predictions are estimates, not guaranteed exam results.
- Model quality depends on the training data and its representativeness.
- Relationships in the dataset do not necessarily imply causation.
- The application is intended for educational and demonstration purposes.
- Predictions should not be used as the sole basis for high-impact academic decisions.

## Future Improvements

- Add a reproducible model evaluation report.
- Compare Ridge Regression with alternative regression models.
- Introduce model explainability and feature analysis.
- Improve input validation and user-facing error handling.
- Add prediction history and downloadable reports.
- Add automated tests for preprocessing and prediction behavior.

## Author

**Abdelrahman Akl**

- GitHub: [@AbdelrhmanAkl](https://github.com/AbdelrhmanAkl)
- Repository: [StudentScore AI](https://github.com/AbdelrhmanAkl/student-score-ai)
- Application: [StudentScore AI Web App](https://student-score-ai.streamlit.app/)

---

If you find this project useful, consider giving the repository a star.

*Built with Python, Machine Learning, and Streamlit.*
