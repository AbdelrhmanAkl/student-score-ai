# 🎓 StudentScore AI
### Machine Learning–Powered Student Exam Score Prediction

**StudentScore AI** is an interactive machine learning web application that estimates student exam scores from academic performance, study habits, lifestyle indicators, and student background factors.

Built with **Python, Scikit-learn, Pandas, Joblib, and Streamlit**, the project demonstrates how a trained regression pipeline can be integrated into a practical, browser-based prediction experience.

<p align="center">
  <a href="https://student-score-ai.streamlit.app/"><strong>🚀 Live Demo</strong></a> •
  <a href="https://github.com/AbdelrhmanAkl/student-score-ai"><strong>💻 GitHub Repository</strong></a>
</p>

---

## 📌 Project Overview

Educational performance is influenced by multiple academic, behavioral, and environmental factors. StudentScore AI provides an interactive way to explore these inputs and generate a model-based estimate of a student's exam score.

Users enter student information through a web interface, and the application passes the input through a saved Scikit-learn pipeline to produce a prediction.

### Key Features

- **Exam Score Prediction** — Estimate exam scores from student-related input features.
- **Academic Performance Analysis** — Include study hours, attendance, previous scores, and tutoring sessions.
- **Lifestyle Factors** — Consider sleep duration and physical activity.
- **Student Background Inputs** — Incorporate selected family, school, and educational resource attributes.
- **End-to-End ML Integration** — Load a trained preprocessing and regression pipeline using Joblib.
- **Interactive Web Experience** — Use the application directly in a browser through Streamlit.
- **Cloud Deployment** — Make the application accessible without local installation.

---

## 🛠️ Technology Stack

| Technology | Role |
|---|---|
| Python | Core programming language |
| Pandas | Data manipulation and input preparation |
| Scikit-learn | Preprocessing and machine learning |
| Ridge Regression | Regression model for score estimation |
| Joblib | Serialization and loading of the trained pipeline |
| Streamlit | Interactive web application |
| Jupyter Notebook | Model development and experimentation |
| Git & GitHub | Version control and source code hosting |
| Streamlit Community Cloud | Application hosting |

---

## 📊 Dataset

The project uses the **Student Performance Factors** dataset, containing **6,607 records and 20 columns**, including the target variable `Exam_Score`.

The dataset includes factors related to:

- Study hours, attendance, and previous academic scores
- Tutoring sessions, sleep, and physical activity
- Motivation and extracurricular activities
- Internet access and educational resources
- Family income and parental education
- Teacher quality, school type, and peer influence

The application uses the dataset to populate categorical input choices. The trained pipeline processes the selected feature values to generate a prediction.

> **Dataset note:** Review the original dataset license and redistribution terms before reusing or redistributing the data.

---

## ⚙️ How It Works

The prediction workflow follows these steps:

1. **User Input** — Collect academic, lifestyle, and background information.
2. **Input Preparation** — Organize the submitted values into a structured Pandas DataFrame.
3. **Preprocessing** — Apply the transformations included in the saved Scikit-learn pipeline.
4. **Prediction** — Generate an estimated exam score using Ridge Regression.
5. **Result Presentation** — Display the predicted score through the Streamlit interface.

The serialized pipeline keeps preprocessing and inference integrated into a single model artifact.

---

## 🧠 Model Architecture

StudentScore AI uses **Ridge Regression**, a regularized linear regression algorithm that applies an L2 penalty to help control model coefficient magnitudes.

The trained model is stored as a serialized Scikit-learn pipeline and loaded by the application at runtime.

### Model Evaluation

For a reliable assessment, regression models should be evaluated on a held-out test set using metrics such as:

| Metric | Purpose |
|---|---|
| MAE | Measures average absolute prediction error |
| RMSE | Penalizes larger prediction errors more strongly |
| R² | Measures how much target variance is explained by the model |

Evaluation results should be reported only when they correspond to the final saved model and a clearly defined test split.

---

## 📁 Project Structure

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

| File or Directory | Description |
|---|---|
| `app.py` | Streamlit application and prediction interface |
| `requirements.txt` | Python dependency definitions |
| `models/` | Saved machine learning model |
| `data/` | Dataset used by the application |
| `notebooks/` | Model exploration, training, and evaluation workflow |
| `.gitignore` | Excludes selected local and generated files |

---

## 🚀 Live Demo

**Try StudentScore AI online:**

👉 https://student-score-ai.streamlit.app/

No local installation is required to access the hosted application.

If the demo is temporarily unavailable, you can clone the repository and run it locally using the instructions below.

---

## 💻 Run Locally

### 1. Clone the Repository

```bash
git clone https://github.com/AbdelrhmanAkl/student-score-ai.git
cd student-score-ai
```

### 2. Create a Virtual Environment

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

### 3. Install Dependencies

```bash
python -m pip install --upgrade pip
pip install -r requirements.txt
```

### 4. Launch the Application

```bash
streamlit run app.py
```

Open the local URL printed in the terminal, typically:

`http://localhost:8501`

**Requirements:** Ensure the trained model and dataset are present at the paths expected by `app.py`.

---

## ☁️ Deployment

The application is designed for deployment through Streamlit Community Cloud.

1. Push the project to GitHub.
2. Create an application in Streamlit Community Cloud.
3. Select the repository and the `main` branch.
4. Set the main file path to `app.py`.
5. Choose a Python version compatible with the project dependencies.
6. Deploy the application and inspect the logs if startup fails.

### Dependency Compatibility

The serialized model was created using **Scikit-learn 1.6.1**. The deployment environment should use a compatible version because serialized Scikit-learn models are not guaranteed to load correctly across library versions.

The dependency is pinned in `requirements.txt` to reduce the risk of model-loading incompatibilities.

---

## ⚠️ Limitations & Responsible Use

- Predictions are estimates, not guaranteed exam results.
- Model performance depends on training data quality and representativeness.
- Statistical relationships do not necessarily imply causation.
- Predictions may be less reliable for students whose characteristics differ from the training data.
- The application is intended for educational and demonstration purposes.
- Predictions should **not** be used as the sole basis for high-impact academic decisions.

---

## 🔮 Future Improvements

Potential next steps include:

- Reproducible model evaluation and experiment tracking
- Comparison with alternative regression algorithms
- Model explainability and feature contribution analysis
- Automated tests for input validation and prediction behavior
- Improved user-facing error handling
- Prediction history and downloadable reports
- Interactive visualizations of dataset patterns

---

## 👨‍💻 Author

**Abdelrahman Akl**

- **GitHub:** [@AbdelrhmanAkl](https://github.com/AbdelrhmanAkl)
- **Project Repository:** [StudentScore AI](https://github.com/AbdelrhmanAkl/student-score-ai)
- **Live Demo:** [student-score-ai.streamlit.app](https://student-score-ai.streamlit.app/)

---

If you find this project interesting, consider giving the repository a ⭐ star.

**Built with Python, Machine Learning, and Streamlit.**