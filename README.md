Support Vector Machines (SVM) Project

This project demonstrates Support Vector Machines (SVM) for binary classification using the Breast Cancer dataset.
It covers linear and non-linear (RBF) kernels, hyperparameter tuning, and visualization of decision boundaries.

📂 Project Structure
SVM_Project/
│
├── data/
│   └── breast-cancer.csv        # dataset
│
├── src/
│   ├── __init__.py
│   ├── load_data.py             # data loading & preprocessing
│   ├── train_svm.py             # SVM model training
│   ├── evaluation.py            # evaluation functions
│   ├── visualize.py             # PCA & decision boundary plotting
│   └── tune_hyperparams.py      # hyperparameter tuning
│
├── main.py                      # main script to run pipeline
├── requirements.txt             # dependencies
└── README.md                    # documentation

⚙️ Installation

Clone or copy this project.

Install required packages:

pip install -r requirements.txt

🚀 How to Run

Run the main script:

python main.py

🔑 Workflow

Load Data → Reads breast-cancer.csv, splits into train/test, scales features.

Train Models →

Linear SVM (kernel="linear")

RBF SVM (kernel="rbf")

Evaluate Models → Prints accuracy & classification report.

Visualize Decision Boundaries → Uses PCA to reduce data to 2D for plotting.

Hyperparameter Tuning → Uses GridSearchCV to optimize C and gamma.

📊 Outputs

Console: Accuracy & classification report for each model.

Plots: 2D decision boundaries for Linear & RBF kernels.

Best hyperparameters from tuning.

🛠️ Tools Used

Python

NumPy / Pandas → Data handling

Scikit-learn → SVM, preprocessing, evaluation, hyperparameter tuning

Matplotlib / Seaborn → Visualization

📌 Notes

Default target column is "diagnosis" (M=1, B=0).

If your dataset uses a different label column, update target_col in src/load_data.py.