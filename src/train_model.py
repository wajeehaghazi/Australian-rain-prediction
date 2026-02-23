from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix, roc_curve, roc_auc_score
from sklearn.model_selection import GridSearchCV, RandomizedSearchCV
import pandas as pd
import matplotlib.pyplot as plt
import joblib


def train_models(df):

    print("Preparing data for training...")

    y = df["RainTomorrow"]
    X = df.drop(columns=["RainTomorrow"])

    train_mask = (df["Date"].dt.year >= 2007) & (df["Date"].dt.year <= 2015)
    test_mask = (df["Date"].dt.year >= 2016) & (df["Date"].dt.year <= 2017)

    X_train = X[train_mask]
    y_train = y[train_mask]

    X_test = X[test_mask]
    y_test = y[test_mask]

    X_train = X_train.drop(columns=["Date"])
    X_test = X_test.drop(columns=["Date"])

    scaler = StandardScaler()

    num_cols = X_train.select_dtypes(include=["int64", "float64"]).columns

    X_train[num_cols] = scaler.fit_transform(X_train[num_cols])
    X_test[num_cols] = scaler.transform(X_test[num_cols])

    results = {}

    print("\nTuning Logistic Regression...")

    param_grid = {
        "C": [0.01, 0.1, 1, 10],
        "solver": ["lbfgs", "liblinear"],
        "penalty": ["l2"]
    }

    grid = GridSearchCV(
        LogisticRegression(max_iter=1000),
        param_grid,
        cv=5,
        scoring="accuracy",
        n_jobs=-1
    )

    grid.fit(X_train, y_train)

    best_log = grid.best_estimator_
    pred = best_log.predict(X_test)

    log_acc = accuracy_score(y_test, pred)

    print("Best Params:", grid.best_params_)
    print("Accuracy:", log_acc)

    results["Logistic Regression"] = (best_log, log_acc)

    print("\nTuning Decision Tree...")

    param_grid_dt = {
        "max_depth": [5, 10, 15, 20],
        "min_samples_split": [2, 5, 10],
        "min_samples_leaf": [1, 2, 4],
        "criterion": ["gini", "entropy"]
    }

    grid_dt = GridSearchCV(
        DecisionTreeClassifier(random_state=42),
        param_grid_dt,
        cv=5,
        scoring="accuracy",
        n_jobs=-1
    )

    grid_dt.fit(X_train, y_train)

    best_dt = grid_dt.best_estimator_
    pred = best_dt.predict(X_test)

    dt_acc = accuracy_score(y_test, pred)

    print("Best Params:", grid_dt.best_params_)
    print("Accuracy:", dt_acc)

    results["Decision Tree"] = (best_dt, dt_acc)

    print("\nTuning Random Forest...")

    param_dist = {
        "n_estimators": [100, 200],
        "max_depth": [10, 20, None],
        "min_samples_split": [2, 5],
        "min_samples_leaf": [1, 2],
        "max_features": ["sqrt", "log2"]
    }

    random_search = RandomizedSearchCV(
        RandomForestClassifier(random_state=42),
        param_distributions=param_dist,
        n_iter=8,
        cv=3,
        scoring="accuracy",
        n_jobs=-1,
        verbose=1
    )

    random_search.fit(X_train, y_train)

    best_rf = random_search.best_estimator_
    pred = best_rf.predict(X_test)

    rf_acc = accuracy_score(y_test, pred)

    print("Best Params:", random_search.best_params_)
    print("Accuracy:", rf_acc)

    results["Random Forest"] = (best_rf, rf_acc)

    print("\nModel Comparison")

    comparison = pd.DataFrame({
        "Model": results.keys(),
        "Accuracy": [results[m][1] for m in results]
    })

    print(comparison.sort_values(by="Accuracy", ascending=False))

    best_model_name = max(results, key=lambda x: results[x][1])
    best_model = results[best_model_name][0]

    print("\nBest Model:", best_model_name)

    joblib.dump(best_model, "models/best_rain_model.pkl")

    print("Model saved to models/best_rain_model.pkl")

    final_pred = best_model.predict(X_test)

    print("\nConfusion Matrix")
    print(confusion_matrix(y_test, final_pred))

    print("\nClassification Report")
    print(classification_report(y_test, final_pred))

    if best_model_name in ["Random Forest", "Decision Tree"]:

        importance = best_model.feature_importances_
        features = X_train.columns

        feat_imp = pd.Series(importance, index=features).sort_values(ascending=False)

        plt.figure(figsize=(10,6))
        feat_imp.head(10).plot(kind="bar")
        plt.title("Top 10 Important Features")
        plt.show()

    if hasattr(best_model, "predict_proba"):

        probs = best_model.predict_proba(X_test)[:,1]

        fpr, tpr, _ = roc_curve(y_test, probs)
        auc_score = roc_auc_score(y_test, probs)

        plt.figure(figsize=(6,6))
        plt.plot(fpr, tpr)
        plt.plot([0,1],[0,1])
        plt.title(f"ROC Curve (AUC = {auc_score:.3f})")
        plt.show()