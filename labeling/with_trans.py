# hugg_min_binary.py
import pandas as pd
from transformers import AutoTokenizer, AutoModelForSequenceClassification, TextClassificationPipeline

# 1) Lire les commentaires
#   - comments_only.csv avec colonne "commentaire"
#   - sinon comments.csv avec "author","text"
try:
    df = pd.read_csv("data/comments.csv", encoding="utf-8")
    df["author"] = df["author"].astype(str)
    df["text"]   = df["text"].astype(str)
    df["commentaire"] = df["author"].fillna("") + " " + df["text"].fillna("")
    df = df[["commentaire"]].copy()
    texts = df["commentaire"].tolist()
except FileNotFoundError:
    df = pd.read_csv("data/comments_only.csv", encoding="utf-8")
    df["commentaire"] = df["text"].astype(str)
    texts = df["commentaire"].tolist()

MODEL = "cardiffnlp/twitter-xlm-roberta-base-sentiment"

# 2) Charger le modèle (préférer safetensors) et le tokenizer "slow" pour éviter les soucis
tok = AutoTokenizer.from_pretrained(MODEL, use_fast=False)
mdl = AutoModelForSequenceClassification.from_pretrained(MODEL, trust_remote_code=False)

pipe = TextClassificationPipeline(
    model=mdl,
    tokenizer=tok,
    truncation=True,        # évite l'avertissement
    max_length=256,         # coupe proprement
    batch_size=64,          # ajuste selon ta RAM CPU
                 # renvoie un dict {"label","score"}
)

# 3) Inférence
preds = pipe(texts)
labels_bin = [1 if p["label"].lower() == "positive" else 0 for p in preds]

# 4) Attach to df
def best_label(obj):
    if isinstance(obj, list):
        obj = max(obj, key=lambda x: x["score"])
    return obj["label"], float(obj["score"])

labels, scores = zip(*[best_label(p) for p in preds])
df["label_raw"] = labels
df["score"] = scores
df["label_bin"] = (
    df["label_raw"].str.lower()
      .map({"positive": 1, "neutral": 0, "negative": 0})
      .fillna(0).astype(int)
)

# 5) Sauvegarde + résumé
out = "labeling/comments_labeled_trans.csv"
df.to_csv(out, index=False, encoding="utf-8-sig")

n_total = len(df)
n_pos = int((df["label_bin"] == 1).sum())
n_neg = n_total - n_pos
print(f"✅ {n_total} commentaires labellises → {out}")
print(f"   → {n_pos} positifs (1)")
print(f"   → {n_neg} negatifs/neutres (0)")