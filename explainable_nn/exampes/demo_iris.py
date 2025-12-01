from sklearn.datasets import load_iris
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import accuracy_score, confusion_matrix
from explainable_nn.core import ExplainableNeuralNet
from explainable_nn.utils import save_model
from pprint import pprint
import numpy as np


def main():

    iris = load_iris()
    X = iris.data
    y = iris.target

    scaler = StandardScaler()
    X = scaler.fit_transform(X)

    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.25, random_state=42)

    nn = ExplainableNeuralNet([4,6,6,3], learning_rate=0.02)

    for epoch in range(300):
        for xi, yi in zip(X_train, y_train):
            nn.train_step(xi, yi)

    preds = [np.argmax(nn.forward(x)) for x in X_test]
    print("Accuracy:", accuracy_score(y_test, preds))
    print(confusion_matrix(y_test, preds))

    save_model("modelo.pkl", nn, scaler)

    ejemplo = X_test[0]
    log = nn.explain(ejemplo)
    pprint(log)


if __name__ == "__main__":
    main()
