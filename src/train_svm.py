from sklearn.svm import SVC

def train_linear(X_train, y_train, C=1):
    model = SVC(kernel="linear", C=C)
    model.fit(X_train, y_train)
    return model

def train_rbf(X_train, y_train, C=1, gamma="scale"):
    model = SVC(kernel="rbf", C=C, gamma=gamma)
    model.fit(X_train, y_train)
    return model
