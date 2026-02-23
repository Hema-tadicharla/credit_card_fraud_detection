# ==============================
# CREDIT CARD FRAUD DETECTION PROJECT
# ==============================

# 1️⃣ Import Libraries
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import (
    accuracy_score,
    classification_report,
    confusion_matrix,
    roc_auc_score,
    roc_curve
)

# 2️⃣ Load Dataset
data = pd.read_csv(r"C:\Users\tadic\Downloads\credit_card_fraud_10k.csv")

print("Dataset Shape:", data.shape)
print("\nFirst 5 Rows:\n", data.head())

# 3️⃣ Check Class Distribution
print("\nClass Distribution:\n", data["is_fraud"].value_counts())

plt.figure()
sns.countplot(x="is_fraud", data=data)
plt.title("Fraud vs Normal Transactions")
plt.show()

# 4️⃣ Drop Unnecessary Column
data = data.drop(columns=["transaction_id"])

# 5️⃣ Encode Categorical Column
le = LabelEncoder()
data["merchant_category"] = le.fit_transform(data["merchant_category"])

# 6️⃣ Separate Features and Target
X = data.drop(columns=["is_fraud"])
y = data["is_fraud"]

# 7️⃣ Train-Test Split
X_train, X_test, y_train, y_test = train_test_split(
    X, y,
    test_size=0.2,
    random_state=42,
    stratify=y
)

# 8️⃣ Feature Scaling
scaler = StandardScaler()
X_train = scaler.fit_transform(X_train)
X_test = scaler.transform(X_test)

# ==================================
# MODEL 1️⃣ Logistic Regression
# ==================================

log_model = LogisticRegression(max_iter=2000)
log_model.fit(X_train, y_train)

y_pred_log = log_model.predict(X_test)

print("\n===== Logistic Regression Results =====")
print("Accuracy:", accuracy_score(y_test, y_pred_log))
print("\nConfusion Matrix:\n", confusion_matrix(y_test, y_pred_log))
print("\nClassification Report:\n", classification_report(y_test, y_pred_log))

# ROC-AUC
log_auc = roc_auc_score(y_test, log_model.predict_proba(X_test)[:,1])
print("ROC-AUC Score:", log_auc)

# ==================================
# MODEL 2️⃣ Random Forest
# ==================================

rf_model = RandomForestClassifier(
    n_estimators=200,
    random_state=42
)

rf_model.fit(X_train, y_train)

y_pred_rf = rf_model.predict(X_test)

print("\n===== Random Forest Results =====")
print("Accuracy:", accuracy_score(y_test, y_pred_rf))
print("\nConfusion Matrix:\n", confusion_matrix(y_test, y_pred_rf))
print("\nClassification Report:\n", classification_report(y_test, y_pred_rf))

rf_auc = roc_auc_score(y_test, rf_model.predict_proba(X_test)[:,1])
print("ROC-AUC Score:", rf_auc)

# ==================================
# ROC Curve Comparison
# ==================================

fpr_log, tpr_log, _ = roc_curve(y_test, log_model.predict_proba(X_test)[:,1])
fpr_rf, tpr_rf, _ = roc_curve(y_test, rf_model.predict_proba(X_test)[:,1])

plt.figure()
plt.plot(fpr_log, tpr_log, label="Logistic Regression")
plt.plot(fpr_rf, tpr_rf, label="Random Forest")
plt.plot([0,1], [0,1])
plt.xlabel("False Positive Rate")
plt.ylabel("True Positive Rate")
plt.title("ROC Curve Comparison")
plt.legend()
plt.show()

# ==================================
# Feature Importance (Random Forest)
# ==================================

feature_importance = pd.Series(
    rf_model.feature_importances_,
    index=X.columns
).sort_values(ascending=False)

print("\nTop Important Features:\n")
print(feature_importance.head())

plt.figure()
feature_importance.head(5).plot(kind='bar')
plt.title("Top 5 Important Features")
plt.show()

# ==================================
# Final Conclusion
# ==================================

if rf_auc > log_auc:
    print("\nRandom Forest performs better for fraud detection.")
else:
    print("\nLogistic Regression performs better.")