from sklearn.model_selection import GridSearchCV
from sklearn.svm import SVC

def tune_svm(X_train, y_train):
    param_grid = {
        "C": [0.1, 1, 10, 100],
        "gamma": ["scale", 0.01, 0.1, 1, 10],
        "kernel": ["rbf"]
    }
    grid = GridSearchCV(SVC(), param_grid, cv=5, scoring="accuracy")
    grid.fit(X_train, y_train)

    print("\nBest Params:", grid.best_params_)
    print("Best Cross-Validation Score:", grid.best_score_)
    return grid.best_estimator_
