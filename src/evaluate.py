from sklearn.metrics import accuracy_score, classification_report

def evaluate_model(model, X_test, y_test, name="Model"):
    preds = model.predict(X_test)
    print(f"\n{name} Accuracy: {accuracy_score(y_test, preds):.4f}")
    print(f"{name} Report:\n", classification_report(y_test, preds))
