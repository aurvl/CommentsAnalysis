# viz.py
import os
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from wordcloud import WordCloud
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.manifold import TSNE
from sklearn.metrics import confusion_matrix
import numpy as np

IN_FILE = "comments_labeled_binary.csv"  # ou comments_labeled.csv selon ton dataset
OUT_DIR = "figs"
os.makedirs(OUT_DIR, exist_ok=True)

# 1) Lecture
df = pd.read_csv(IN_FILE, sep=";", encoding="utf-8-sig")
df["commentaire"] = df["commentaire"].astype(str).str.strip()
df["label"] = df["label"].astype(int)

# 2) TF-IDF (mêmes paramètres que tp_min_models.py)
vectorizer = TfidfVectorizer(analyzer="word", lowercase=True, ngram_range=(1,2), min_df=2, max_df=0.9)
X = vectorizer.fit_transform(df["commentaire"].values)
vocab = np.array(vectorizer.get_feature_names_out())

# 3) Wordclouds par classe
for label in sorted(df["label"].unique()):
    texts = " ".join(df.loc[df["label"]==label, "commentaire"].values)
    wc = WordCloud(width=1200, height=600, background_color="white").generate(texts)
    plt.figure(figsize=(12,6))
    plt.imshow(wc, interpolation="bilinear")
    plt.axis("off")
    plt.title(f"Nuage de mots - label={label}")
    plt.tight_layout()
    plt.savefig(os.path.join(OUT_DIR, f"wordcloud_label_{label}.png"))
    plt.close()

# 4) Top mots TF-IDF globaux (somme des poids)
tfidf_sum = np.asarray(X.sum(axis=0)).ravel()
top_idx = tfidf_sum.argsort()[::-1][:30]
plt.figure(figsize=(10,6))
sns.barplot(x=tfidf_sum[top_idx], y=vocab[top_idx])
plt.title("Top 30 mots (somme TF-IDF)")
plt.tight_layout()
plt.savefig(os.path.join(OUT_DIR, "top_tfidf.png"))
plt.close()

# 5) Top coefficients de la Logistic Regression (entraîne sur tout le dataset)
log = LogisticRegression(penalty="l2", C=1.0, max_iter=2000)
log.fit(X, df["label"].values)
coefs = log.coef_.ravel()
top_pos = np.argsort(coefs)[-20:][::-1]
top_neg = np.argsort(coefs)[:20]

plt.figure(figsize=(10,6))
sns.barplot(x=coefs[top_pos], y=vocab[top_pos], color="green")
plt.title("Top 20 mots positifs (coef LogReg)")
plt.tight_layout()
plt.savefig(os.path.join(OUT_DIR, "top_logreg_pos.png"))
plt.close()

plt.figure(figsize=(10,6))
sns.barplot(x=coefs[top_neg], y=vocab[top_neg], color="red")
plt.title("Top 20 mots négatifs (coef LogReg)")
plt.tight_layout()
plt.savefig(os.path.join(OUT_DIR, "top_logreg_neg.png"))
plt.close()

# 6) t-SNE des documents (usage recommandé sur un échantillon si dataset grand)
sample_size = min(1000, X.shape[0])
idx = np.random.RandomState(42).choice(X.shape[0], sample_size, replace=False)
X_sample = X[idx].toarray()
y_sample = df["label"].values[idx]

tsne = TSNE(n_components=2, random_state=42, init="pca", learning_rate="auto")
X2 = tsne.fit_transform(X_sample)

plt.figure(figsize=(8,6))
palette = {0: "red", 1: "green"}
sns.scatterplot(x=X2[:,0], y=X2[:,1], hue=y_sample, palette=palette, s=40, alpha=0.8)
plt.title("t-SNE (échantillon)")
plt.legend(title="label")
plt.tight_layout()
plt.savefig(os.path.join(OUT_DIR, "tsne_docs.png"))
plt.close()

print(f"Figures enregistrées dans le dossier: {OUT_DIR}")