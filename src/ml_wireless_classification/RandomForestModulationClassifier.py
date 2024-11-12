# ml_wireless_classification/models/RandomForestModulationClassifier.py

from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score

from ml_wireless_classification.base.BaseModulationClassifier import BaseModulationClassifier

class RandomForestModulationClassifier(BaseModulationClassifier):
    def __init__(self, n_estimators=100, random_state=42):
        self.model = RandomForestClassifier(n_estimators=n_estimators, random_state=random_state)

    def fit(self, X, y):
        self.model.fit(X, y)

    def predict(self, X):
        return self.model.predict(X)

    def evaluate(self, X, y):
        predictions = self.predict(X)
        accuracy = accuracy_score(y, predictions)
        return accuracy, predictions
    
    def prepare_data(self):
        return super().prepare_data()
