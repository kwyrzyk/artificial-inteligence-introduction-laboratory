from ucimlrepo import fetch_ucirepo 
import numpy as np
import matplotlib.pyplot as plt
from sklearn.model_selection import KFold
from sklearn.metrics import accuracy_score, confusion_matrix, classification_report
import seaborn as sns
from NaiveBayesClassifier import *
import matplotlib
matplotlib.use("Agg")

def evaluate_classifiers(classifiers, classifier_names, X, y, n_splits=5):
    kf = KFold(n_splits=n_splits, shuffle=True, random_state=42)

    accuracies = {name: [] for name in classifier_names}
    confusion_matrices = {name: np.zeros((len(np.unique(y)), len(np.unique(y))), dtype=int) for name in classifier_names}

    for train_index, test_index in kf.split(X):
        X_train, X_test = X[train_index], X[test_index]
        y_train, y_test = y[train_index], y[test_index]

        for classifier, name in zip(classifiers, classifier_names):
            classifier.fit(X_train, y_train)
            y_pred = classifier.predict(X_test)
            accuracy = accuracy_score(y_test, y_pred)
            accuracies[name].append(accuracy)
            confusion_matrices[name] += confusion_matrix(y_test, y_pred)

    plt.figure(figsize=(10, 6))
    for name in classifier_names:
        plt.plot(range(1, n_splits + 1), accuracies[name], marker='o', linestyle='--', label=name)
    plt.xlabel('Fold', fontweight="bold")
    plt.ylabel('Accuracy', fontweight="bold")
    plt.title('Accuracy in each fold by classifier', fontweight="bold")
    min_acc = min(min(acc) for acc in accuracies.values())
    max_acc = max(max(acc) for acc in accuracies.values())
    plt.ylim(min_acc * 0.95, max_acc*1.05)
    plt.grid(True, linestyle='--', alpha=0.7)
    plt.legend()
    plt.savefig("accuracy_in_each_fold.png")

    plt.figure(figsize=(10, 6))
    for name in classifier_names:
        avg_accuracy = np.mean(accuracies[name])
        plt.scatter([name], [avg_accuracy], color='red', s=100, edgecolor='black', zorder=5)
        plt.text(name, avg_accuracy + 0.01, f'{avg_accuracy:.2f}', va='center', ha='left', color='red')
    plt.xlabel('Classifier', fontweight="bold")
    plt.ylabel('Average Accuracy', fontweight="bold")
    plt.title('Average Accuracy by Classifier', fontweight="bold")
    plt.ylim(min_acc * 0.95, 1.05)
    plt.grid(True, linestyle='--', alpha=0.7)
    plt.savefig("average_accuracy_comparison.png")

    for name, matrix in confusion_matrices.items():
        plt.figure(figsize=(8, 6))
        sns.heatmap(matrix, annot=True, fmt="d", cmap="Blues", xticklabels=np.unique(y), yticklabels=np.unique(y))
        plt.title(f"Confusion Matrix (Cumulative, 5-fold CV): {name}")
        plt.xlabel("Predicted")
        plt.ylabel("Actual")
        plt.savefig(f"confusion_matrix_{name}.png")

if __name__ == "__main__":
    from sklearn.tree import DecisionTreeClassifier
    from sklearn.neighbors import KNeighborsClassifier

    iris = fetch_ucirepo(id=53) 
    
    X = np.array(iris.data.features)
    y = np.array(iris.data.targets).flatten()

    classifiers = [NaiveBayesClassifier(GaussianPDF()), DecisionTreeClassifier(), KNeighborsClassifier()]
    classifier_names = ["Naive Bayes", "Decision Tree", "K-Nearest Neighbors"]

    evaluate_classifiers(classifiers, classifier_names, X, y)
