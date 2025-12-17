from sklearn.linear_model import LogisticRegression
from sklearn.neighbors import KNeighborsClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.svm import SVC
from lightgbm import LGBMClassifier
from catboost import CatBoostClassifier
from sklearn.metrics import accuracy_score

def get_models():
    models = {
        "LogisticRegression": LogisticRegression(random_state=12345),
        "KNN": KNeighborsClassifier(),
        "DecisionTree": DecisionTreeClassifier(random_state=12345),
        "RandomForest": RandomForestClassifier(random_state=12345),
        "SVM": SVC(gamma="auto", probability=True, random_state=12345),
        "GradientBoosting": GradientBoostingClassifier(random_state=12345),
        "LightGBM": LGBMClassifier(random_state=12345),
        "CatBoost": CatBoostClassifier(random_state=12345, verbose=False)
    }
    return models

def train_and_evaluate(models, X_train, X_test, y_train, y_test):
    results = {}
    for name, model in models.items():
        model.fit(X_train, y_train)
        preds = model.predict(X_test)
        acc = accuracy_score(y_test, preds)
        results[name] = acc
    return results
