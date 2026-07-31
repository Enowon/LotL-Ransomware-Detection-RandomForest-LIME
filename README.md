# AI-Assisted Detection of Living-off-the-Land Ransomware Attacks
## Using Random Forest and LIME Explainability on Windows Sysmon Logs

![Python](https://img.shields.io/badge/Python-3.13.5-blue?style=flat&logo=python)
![Scikit-learn](https://img.shields.io/badge/Scikit--learn-1.9.0-orange?style=flat&logo=scikit-learn)
![LIME](https://img.shields.io/badge/LIME-0.2.0.1-green?style=flat)
![Pandas](https://img.shields.io/badge/Pandas-3.0.3-purple?style=flat&logo=pandas)
![Matplotlib](https://img.shields.io/badge/Matplotlib-3.11.0-red?style=flat)
![University](https://img.shields.io/badge/University-Roehampton-darkblue?style=flat)
![Status](https://img.shields.io/badge/Status-Complete-brightgreen?style=flat)
![License](https://img.shields.io/badge/License-MIT-yellow?style=flat)

**MSc Cybersecurity Project — University of Roehampton, 2026**

---

## Project Overview

This project develops and evaluates an AI-assisted detection system for 
identifying Living-off-the-Land (LotL) ransomware attacks in Windows Sysmon 
log data. LotL attacks abuse legitimate, pre-installed Windows tools such as 
PowerShell and WMI to carry out malicious activity, leaving no external files 
to scan and no signatures to match. This makes them exceptionally difficult to 
detect using conventional rule-based approaches.

The system uses a Random Forest machine learning classifier trained on the 
CSU Ransomware Dataset, compared against Decision Tree and Logistic Regression 
as baseline classifiers, and evaluated against a feature-based Sigma rule 
approximation. LIME (Local Interpretable Model-Agnostic Explanations) is applied 
to generate interpretable, feature-level explanations for individual model 
predictions, making the system practical and transparent for SOC analysts.

---

## Key Results

| Model | Accuracy | Precision | Recall | F1 Score |
|---|---|---|---|---|
| Random Forest | 98.15% | 93.04% | 99.37% | 96.10% |
| Decision Tree | 98.12% | 93.03% | 99.23% | 96.03% |
| Logistic Regression | 95.59% | 89.86% | 91.02% | 90.43% |
| Sigma Rules (Strict) | 77.12% | 85.34% | 0.20% | 0.40% |
| Sigma Rules (Loosened) | 75.26% | 44.46% | 31.95% | 37.18% |

---

## Dataset

The primary dataset used is the **CSU Ransomware Dataset**, publicly available at:

https://github.com/CSCRC-SCREED/CSU-Ransomware-Data

- 352,876 labelled Windows Sysmon log entries
- 19 pre-engineered behavioural features
- Binary labels: benign (good) and ransomware (ransom)

---

## Project Structure

├── explore_data.py # Load and explore the dataset
├── preprocess.py # Label encoding, feature split, train/test split
├── train_model.py # Train the Random Forest classifier
├── evaluate_model.py # Evaluate model performance and confusion matrix
├── model_comparison.py # Compare Random Forest, Decision Tree, Logistic Regression
├── model_comparison_cv.py # Cross-validation comparison of all three models
├── hyperparameter_tuning.py # Grid Search hyperparameter optimisation
├── sigma_rules.py # Feature-based Sigma rule baseline (strict)
├── sigma_rules_updated.py # Sigma rule sensitivity analysis (loosened)
├── lime_explain.py # LIME explainability for individual predictions
├── feature_importance.py # Feature importance analysis and visualisation
└── requirements.txt # Required Python libraries


---

## Requirements

- Python 3.13.5
- See requirements.txt for all required libraries

To install all required libraries, run:

pip install -r requirements.txt


---

## How to Run

1. Clone this repository or download the files
2. Download the CSU Ransomware Dataset from the link above and save it as 
   `Ransomware_Data.csv` in the same folder as the scripts
3. Install the required libraries using the command above
4. Run the scripts in this order:

python explore_data.py
python preprocess.py
python train_model.py
python evaluate_model.py
python model_comparison.py
python model_comparison_cv.py
python hyperparameter_tuning.py
python sigma_rules.py
python sigma_rules_updated.py
python lime_explain.py
python feature_importance.py


---

## Technologies Used

| Tool | Purpose |
|---|---|
| Python 3.13.5 | Primary programming language |
| Pandas 3.0.3 | Data loading and manipulation |
| Scikit-learn 1.9.0 | Model training and evaluation |
| LIME 0.2.0.1 | Explainability |
| Matplotlib 3.11.0 | Data visualisation |

---

## Author

**Faith Okonoboh**
MSc Cybersecurity
University of Roehampton
2026

---

## Acknowledgements

This project was supervised by Dr Mastaneh Davis, University of Roehampton.
The CSU Ransomware Dataset was created by CSCRC-SCREED and is used here 
for academic research purposes only.

---

## Licence

This project is licensed under the MIT Licence — see the LICENSE file for details.
