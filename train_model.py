
from pathlib import Path
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.svm import SVC
from sklearn.metrics import (classification_report, confusion_matrix, accuracy_score)
import pickle

# 1. Load Data
DATA_PATH = Path("Manufacturing_dataset.xls")
if not DATA_PATH.exists():
    raise FileNotFoundError(f"{DATA_PATH} not found.")

df = pd.read_csv(DATA_PATH)
print("Dataset shape:", df.shape)

# 2. Balance Data
target_col = 'Optimal Conditions'
df0 = df[df[target_col] == 0]
df1 = df[df[target_col] == 1]

print(f"Total Class 0: {len(df0)}")
print(f"Total Class 1: {len(df1)}")

# Apply sampling strategy: 1500 from Class 0, ALL from Class 1
df0_sample = df0.sample(n=1500, random_state=42)   # reduce class 0
df1_sample = df1                                   # keep all class 1

df_balanced = pd.concat([df0_sample, df1_sample])
df_balanced = df_balanced.sample(frac=1, random_state=42).reset_index(drop=True)
df1500 = df_balanced

print("Balanced dataset shape:", df1500.shape)
print("Class distribution in balanced data:")
print(df1500[target_col].value_counts())

# 3. Preprocess
X = df1500.drop(columns=[target_col])
if 'Timestamp' in X.columns:
    X = X.drop(columns=['Timestamp'])
    print("Removed Timestamp column")
y = df1500[target_col]

if y.dtype == object or y.dtype.name == "category":
    le = LabelEncoder()
    y_enc = le.fit_transform(y.astype(str))
    target_mapping = dict(zip(le.classes_, range(len(le.classes_))))
else:
    y_enc = y.values
    le = None
    target_mapping = None

num_cols = X.select_dtypes(include=[np.number]).columns.tolist()
cat_cols = X.select_dtypes(exclude=[np.number]).columns.tolist()

for c in num_cols:
    med = X[c].median()
    X[c] = X[c].fillna(med)

if cat_cols:
    X_cat = pd.get_dummies(X[cat_cols].astype(str), dummy_na=True, drop_first=True)
    X_num = X[num_cols].reset_index(drop=True)
    X_pre = pd.concat([X_num, X_cat.reset_index(drop=True)], axis=1)
else:
    X_pre = X[num_cols].copy()

# 4. Split
try:
    X_train, X_test, y_train, y_test = train_test_split(X_pre, y_enc, test_size=0.33, random_state=42, stratify=y_enc)
except Exception:
    X_train, X_test, y_train, y_test = train_test_split(X_pre, y_enc, test_size=0.33, random_state=42)

scaler = StandardScaler()
X_train_s = scaler.fit_transform(X_train)
X_test_s = scaler.transform(X_test)

# 5. Train
print("Training SVM...")
param_grid = {
    "C": [0.1, 1, 10, 100],
    "gamma": ["scale", "auto", 0.001, 0.01, 0.1],
    "kernel": ["rbf"],
    "class_weight": ["balanced"]
}
svc = SVC(probability=True, random_state=42)
grid = GridSearchCV(svc, param_grid, cv=5, n_jobs=-1, verbose=1, scoring='f1')
grid.fit(X_train_s, y_train)
best = grid.best_estimator_

print("Best params:", grid.best_params_)
print(f"Best CV Score: {grid.best_score_:.4f}")

# 6. Evaluate
y_pred = best.predict(X_test_s)
acc = accuracy_score(y_test, y_pred)
cm = confusion_matrix(y_test, y_pred)
n_classes = cm.shape[0]

print(f"Test Accuracy: {acc:.4f}")

# 7. Save
model_out = {
    "model": best,
    "scaler": scaler,
    "label_encoder": le,
    "target_mapping": target_mapping,
    "features": X_pre.columns.tolist(),
    "test_accuracy": acc,
    "best_params": grid.best_params_,
    "cv_score": grid.best_score_,
    "n_classes": n_classes
}

out_path = "svm_model.pkl"
with open(out_path, "wb") as file:
    pickle.dump(model_out, file)
    
print(f"Model saved to: {out_path}")
