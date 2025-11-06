# === Imports des libs ===
from sklearn.feature_extraction.text import CountVectorizer   # pr transformer du txt en chiffres (sac de mots)
from sklearn.model_selection import train_test_split          # pr couper en train/test
from sklearn.preprocessing import LabelEncoder                # pr changer "pos"/"neg" en 0/1
import numpy as np
from sklearn.tree import DecisionTreeClassifier               # arbre de decision
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix  # pr evaluer
from sklearn.naive_bayes import MultinomialNB                 # le Naive Bayes (classik pr txt)
from sklearn.linear_model import LogisticRegression


# --- Chargement data ---
donnees = []         # les phrases
sent_donnees = []    # les labels (pos/neg)

# pos : on met chaque phrase + le label pos
for line in open("rt-polarity-pos.txt", "r", encoding="latin-1"):
    donnees.append(line.strip())    # vire les espaces fin
    sent_donnees.append("pos")

# neg : pareil ms avec neg
for line in open("rt-polarity-neg.txt", "r", encoding="latin-1"):
    donnees.append(line.strip())
    sent_donnees.append("neg")

# --- Vectorizer ---
# en gros ça choppe tt les mots et fait un gros tableau avec des 0/1
vectorizer = CountVectorizer(analyzer='word', lowercase=False)
features = vectorizer.fit_transform(donnees)   # fit sur tt les phrases (ms c juste pr avoir une idée)
features_array = features.toarray()            # on converti en array (dense) ms on va pas trop l’utiliser

# --- Encodage labels ---
# change pos/neg en 0/1 sinon les models pigent pas
le = LabelEncoder()
y = le.fit_transform(sent_donnees)   # "neg"=0 et "pos"=1

# --- Split train/test ---
# coupe en 80% pr entrainer et 20% pr tester
# stratify=y = pr garder meme proportion pos/neg
donnees_train, donnees_test, y_train, y_test = train_test_split(
    donnees, y,
    test_size=0.2,
    random_state=42,
    stratify=y
)

# --- Refit du vectorizer ---
# on fit que sur le train (sinon fuite de data)
X_train = vectorizer.fit_transform(donnees_train)
X_test  = vectorizer.transform(donnees_test)

# --- Arbre de decision ---
clf = DecisionTreeClassifier(random_state=42)  # on cree l’arbre
clf.fit(X_train, y_train)                      # on entraine
y_pred = clf.predict(X_test)                   # on test sur le test set

# --- Affiche 15 phrases + pred ---
for i in range(15):
    print(f"Phrase : {donnees_test[i]}")
    print("Prédiction :", "positif" if y_pred[i] == 1 else "négatif")
    print("-" * 80)

# --- Eval arbre ---
print("Accuracy :", accuracy_score(y_test, y_pred))
print("\nRapport de classification :\n", classification_report(y_test, y_pred))
print("\nMatrice de confusion :\n", confusion_matrix(y_test, y_pred))

# --- Naive Bayes ---
# model bcp utilisé pr txt (souvent + fort que arbre simple)
nb = MultinomialNB()
nb.fit(X_train, y_train)
y_pred_nb = nb.predict(X_test)

print("Naive Bayes - Accuracy :", accuracy_score(y_test, y_pred_nb))
print(classification_report(y_test, y_pred_nb))

# --- Regression Logistique ---

# model RL : cost fct logloss, bien pr texte (linéaire + rapide)
# max_iter pr eviter warning de convergence
# +grand = -de régul (peut overfit), +petit = +régul
lr = LogisticRegression(max_iter=2000, C=1.0, random_state=42)

# entrainement
lr.fit(X_train, y_train)

# pred
y_pred_lr = lr.predict(X_test)

# eval
print("Logistic Regression - Accuracy :", accuracy_score(y_test, y_pred_lr))
print(classification_report(y_test, y_pred_lr))