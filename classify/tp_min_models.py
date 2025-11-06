"""
tp_min_models.py
- Lecture du dataset comments_labeled_binary.csv (commentaire;label)
- Vectorisation TF-IDF (1-2 grammes)
- Entrainement: DecisionTree, LogisticRegression (L2), LinearSVC
- Sauvegarde: tp_min_results.csv, tp_test_predictions_tree.csv
- Affichage: top-mots pro/anti de la Logistic Regression
"""

import numpy as np
import pandas as pd
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.model_selection import train_test_split
from sklearn.tree import DecisionTreeClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.svm import LinearSVC
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix
from sklearn.feature_extraction.text import TfidfVectorizer

IN_FILE = "comments_labeled_binary.csv"   # colonnes: commentaire ; label

# 1) Lecture robuste (BOM) + contrôle colonnes
df = pd.read_csv(IN_FILE, sep=";", encoding="utf-8-sig")
df.columns = [c.replace("\ufeff","").strip() for c in df.columns]
if not {"commentaire","label"}.issubset(df.columns):
    raise SystemExit(f"Le fichier doit contenir 'commentaire' et 'label'. Colonnes lues: {list(df.columns)}")

df["commentaire"] = df["commentaire"].astype(str).str.strip()
df["label"] = df["label"].astype(int)

X_text = df["commentaire"].values
y = df["label"].values

# 2) Vectorisation sac-de-mots (comme dans le TP)

vectorizer = TfidfVectorizer(analyzer="word",
                             lowercase=True,
                             ngram_range=(1,2),   # uni + bi-grammes
                             min_df=2,
                             max_df=0.9)
X = vectorizer.fit_transform(X_text)

# 3) Split 80/20 reproductible
X_train, X_test, y_train, y_test, txt_train, txt_test = train_test_split(
    X, y, df["commentaire"].values, test_size=0.2, random_state=42, stratify=y
)

# 4) Trois modèles
models = {
    "DecisionTree": DecisionTreeClassifier(random_state=42),
    "LogisticRegression": LogisticRegression(penalty="l2", C=1.0, max_iter=2000),
    "LinearSVC": LinearSVC(C=1.0, max_iter=5000),
}

results = []
for name, model in models.items():
    print(f"\n=== {name} ===")
    model.fit(X_train, y_train)
    y_pred = model.predict(X_test)
    acc = accuracy_score(y_test, y_pred)
    print(f"Accuracy = {acc:.4f}")
    print("Confusion matrix:"); print(confusion_matrix(y_test, y_pred))
    print("Classification report:"); print(classification_report(y_test, y_pred, digits=4))
    results.append((name, acc))

# 5) Sauvegarde résultats + 15 exemples pour l'Arbre (comme le TP)
pd.DataFrame(results, columns=["model","accuracy"]).to_csv(
    "tp_min_results.csv", index=False, sep=";", encoding="utf-8-sig"
)

tree = models["DecisionTree"]
y_pred_tree = tree.predict(X_test)
out = pd.DataFrame({"commentaire": txt_test, "y_true": y_test, "y_pred_tree": y_pred_tree})
out.to_csv("tp_test_predictions_tree.csv", index=False, sep=";", encoding="utf-8-sig")

print("\n[OK] Résultats enregistrés: tp_min_results.csv, tp_test_predictions_tree.csv")
