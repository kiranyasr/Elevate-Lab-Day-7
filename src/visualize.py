import numpy as np
import matplotlib.pyplot as plt
from sklearn.decomposition import PCA
from sklearn.svm import SVC

def visualize_decision_boundary(X_train, y_train, kernel="rbf"):
    # Reduce to 2D
    pca = PCA(n_components=2)
    X_train_pca = pca.fit_transform(X_train)

    # Train SVM on PCA data
    clf = SVC(kernel=kernel, C=1, gamma="scale")
    clf.fit(X_train_pca, y_train)

    # Meshgrid for plotting
    x_min, x_max = X_train_pca[:, 0].min() - 1, X_train_pca[:, 0].max() + 1
    y_min, y_max = X_train_pca[:, 1].min() - 1, X_train_pca[:, 1].max() + 1
    xx, yy = np.meshgrid(np.arange(x_min, x_max, 0.02),
                         np.arange(y_min, y_max, 0.02))

    # Predictions
    Z = clf.predict(np.c_[xx.ravel(), yy.ravel()])
    Z = Z.reshape(xx.shape)

    # Plot
    plt.contourf(xx, yy, Z, alpha=0.3)
    plt.scatter(X_train_pca[:, 0], X_train_pca[:, 1], c=y_train, edgecolors="k")
    plt.title(f"SVM Decision Boundary ({kernel} kernel)")
    plt.show()
