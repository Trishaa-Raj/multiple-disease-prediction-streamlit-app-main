# 🏥 Multiple Disease Prediction System

A machine learning web application built with **Streamlit** that predicts the likelihood of three diseases — **Diabetes**, **Heart Disease**, and **Parkinson's Disease** — based on medical input values entered by the user.

> ⚠️ **Disclaimer:** This tool is built for educational and learning purposes only. It is **not** a substitute for professional medical advice, diagnosis, or treatment. Always consult a qualified healthcare provider.

---

## 📌 What This Project Does

The app takes medical measurements as input (like blood sugar levels, heart rate, or voice recordings) and uses pre-trained machine learning models to predict whether a person is likely to have a particular disease. Results are shown instantly with color-coded output and a confidence score.

---

## 🖥️ Live Features

- **Three disease modules** accessible from a sidebar navigation menu
- **Real-time prediction** with confidence percentage
- **Color-coded results** — red for positive (disease detected), green for negative (no disease)
- **Input hints** showing normal ranges for each field to guide the user
- **Input validation** — alerts if a field is left empty or contains non-numeric values
- **Reset button** to clear all inputs and start fresh
- **Disclaimer** shown in the sidebar on every page

---

## 🧬 Diseases Covered

### 1. 🩸 Diabetes Prediction
- **Model:** Support Vector Machine (SVM) with linear kernel
- **Dataset:** PIMA Indians Diabetes Dataset — 768 patients, 8 features
- **Input features:** Number of pregnancies, glucose level, blood pressure, skin thickness, insulin level, BMI, diabetes pedigree function, age
- **Model accuracy:** ~77% on test data

### 2. ❤️ Heart Disease Prediction
- **Model:** Logistic Regression
- **Dataset:** Cleveland Heart Disease Dataset — 303 patients, 13 features
- **Input features:** Age, sex, chest pain type, resting blood pressure, cholesterol, fasting blood sugar, resting ECG results, max heart rate, exercise-induced angina, ST depression (oldpeak), slope of ST segment, number of major vessels, thalassemia type
- **Model accuracy:** ~82% on test data

### 3. 🧠 Parkinson's Disease Prediction
- **Model:** Support Vector Machine (SVM) with linear kernel
- **Dataset:** UCI Parkinson's Dataset — 195 voice recordings, 22 features
- **Input features:** 22 voice measurement parameters including MDVP frequency values, jitter, shimmer, noise-to-harmonics ratio, and nonlinear dynamical complexity measures (RPDE, DFA, D2, PPE)
- **Model accuracy:** ~87% on test data

---

## 🗂️ Project Structure

```
multiple-disease-prediction-streamlit-app/
│
├── app.py                          # Main Streamlit application
├── requirements.txt                # Python dependencies
│
├── saved_models/                   # Pre-trained ML models (pickle files)
│   ├── diabetes_model.sav
│   ├── heart_disease_model.sav
│   └── parkinsons_model.sav
│
├── dataset/                        # Raw datasets used for training
│   ├── diabetes.csv
│   ├── heart.csv
│   └── parkinsons.csv
│
└── colab_files_to_train_models/    # Jupyter notebooks (Google Colab)
    ├── Multiple disease prediction system - diabetes.ipynb
    ├── Multiple disease prediction system - heart.ipynb
    └── Multiple disease prediction system - Parkinsons.ipynb
```

---

## 🛠️ Tech Stack

| Purpose | Tool / Library |
|---|---|
| Web app framework | [Streamlit](https://streamlit.io/) 1.29.0 |
| Sidebar navigation | [streamlit-option-menu](https://github.com/victoryhb/streamlit-option-menu) 0.3.6 |
| Machine learning | [scikit-learn](https://scikit-learn.org/) 1.3.2 |
| Numerical computation | [NumPy](https://numpy.org/) 1.26.3 |
| Model training environment | Google Colab (Jupyter Notebooks) |
| Model serialization | Python `pickle` |
| Styling | Custom CSS injected via `st.markdown()` |

---

## ⚙️ How the Models Were Built

All three models were trained inside Google Colab notebooks (included in the `colab_files_to_train_models/` folder). The general training workflow for each disease was:

1. Load the CSV dataset using pandas
2. Separate features (X) and labels (Y)
3. Standardize features using `StandardScaler` (Parkinson's only)
4. Split data into training (80%) and test (20%) sets using `train_test_split`
5. Train the model (SVM or Logistic Regression)
6. Evaluate accuracy on both training and test sets
7. Save the trained model as a `.sav` file using `pickle`

The saved `.sav` files are loaded at app startup using `@st.cache_resource` so they are loaded only once per session.

---

## 📦 Requirements

```
numpy==1.26.3
scikit-learn==1.3.2
streamlit==1.29.0
streamlit-option-menu==0.3.6
```

> **Note:** Additional libraries (pandas, matplotlib, etc.) may be needed to re-run the Jupyter training notebooks.

---

## 📊 Datasets

| Disease | Source | Samples | Features |
|---|---|---|---|
| Diabetes | PIMA Indians Diabetes Database | 768 | 8 |
| Heart Disease | Cleveland Heart Disease (UCI ML Repository) | 303 | 13 |
| Parkinson's | UCI Parkinson's Dataset (voice recordings) | 195 | 22 |

All datasets are included in the `dataset/` folder.

---

## 🔮 How to Retrain the Models

Open the notebooks inside `colab_files_to_train_models/` in Google Colab or Jupyter. Each notebook is self-contained — it loads the dataset, trains the model, and saves the `.sav` file. Replace the corresponding file in `saved_models/` with the newly generated one and restart the app.

---

## 🙋 Who Is This For?

- Students learning machine learning and how to deploy ML models as web apps
- Developers exploring Streamlit for building data science applications
- Anyone curious about how medical datasets can be used for classification problems

---

## 📄 License

This project is open-source and available for educational use.
