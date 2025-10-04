from src.load_data import load_and_preprocess
from src.train_svm import train_linear, train_rbf
from src.evaluate import evaluate_model
from src.visualize import visualize_decision_boundary
from src.tune_hyperparams import tune_svm

def main():
    # 1. Load data
    X_train, X_test, y_train, y_test = load_and_preprocess()

    # 2. Train models
    linear_model = train_linear(X_train, y_train)
    rbf_model = train_rbf(X_train, y_train)

    # 3. Evaluate
    evaluate_model(linear_model, X_test, y_test, "Linear SVM")
    evaluate_model(rbf_model, X_test, y_test, "RBF SVM")

    # 4. Visualize decision boundaries (with PCA)
    visualize_decision_boundary(X_train, y_train, kernel="linear")
    visualize_decision_boundary(X_train, y_train, kernel="rbf")

    # 5. Hyperparameter tuning
    best_model = tune_svm(X_train, y_train)
    evaluate_model(best_model, X_test, y_test, "Tuned RBF SVM")

if __name__ == "__main__":
    main()
