import pandas as pd
import numpy as np
from sklearn.metrics import confusion_matrix, cohen_kappa_score, classification_report
from statsmodels.stats.contingency_tables import mcnemar
import matplotlib.pyplot as plt

# 1. load data
trans = pd.read_csv("labeling/comments_labeled_trans.csv", encoding="utf-8-sig")
lexic = pd.read_csv("labeling/comments_labeled_lexic.csv", encoding="utf-8-sig")

# ensure there is a 'commentaire' column in both
assert "commentaire" in trans.columns, "trans file missing 'commentaire'"
assert "commentaire" in lexic.columns, "lexic file missing 'commentaire'"

trans = trans.rename(columns={
    "label_bin": "label_bin_trans",
    "label_raw": "label_raw_trans",
    "score":     "score_trans"
})

# rename lexic columns -> add _lexic suffixes
if "label_bin" in lexic.columns:
    lexic = lexic.rename(columns={"label_bin": "label_bin_lexic"})
elif "label" in lexic.columns:
    lexic = lexic.rename(columns={"label": "label_bin_lexic"})
else:
    raise KeyError(f"no label column found in lexic: {list(lexic.columns)}")

# keep only needed cols
keep_t = [c for c in ["commentaire","label_bin_trans","label_raw_trans","score_trans"] if c in trans.columns]
keep_l = [c for c in ["commentaire","label_bin_lexic"] if c in lexic.columns]

trans = trans[keep_t].drop_duplicates(subset=["commentaire"])
lexic = lexic[keep_l].drop_duplicates(subset=["commentaire"])

# merge
merged = pd.merge(trans, lexic, on="commentaire", how="inner")
print("Merged data shape:", merged.shape)

def pick(series_names, prefix):
    for name in series_names:
        if name in merged.columns:
            return name
    raise KeyError(f"aucune colonne label pour {prefix} dans: {list(merged.columns)}")

y_trans_col = None
y_lexic_col = None

# probables pour trans
cands_trans = ["label_bin_trans", "label_trans", "pred_trans", "y_trans"]
# probables pour lexic
cands_lexic = ["label_bin_lexic", "label_lexic", "pred_lexic", "y_lexic"]

y_trans_col = pick(cands_trans, "trans")
y_lexic_col = pick(cands_lexic, "lexic")

# optionnels: score du modele trans
score_trans_col = None
for c in ["score_trans", "prob_trans", "p_trans", "confidence_trans", "score"]:
    if c in merged.columns:
        score_trans_col = c
        break

# caster proprement
y_true_trans = merged[y_trans_col].astype(int)
y_true_lexic = merged[y_lexic_col].astype(int)

# retirer les NaN eventuels
mask = y_true_trans.notna() & y_true_lexic.notna()
df = merged.loc[mask].copy()
y_t = df[y_trans_col].astype(int)
y_l = df[y_lexic_col].astype(int)

n = len(df)
print(f"n = {n} lignes comparees")

# 2) accord & statistiques
acc = (y_t == y_l).mean()
cm = confusion_matrix(y_l, y_t, labels=[0,1])  # lignes: lexic, colonnes: trans
kappa = cohen_kappa_score(y_l, y_t)

print("\n--- accord global")
print(f"taux d’accord: {acc:.3f}")
print(f"cohen’s kappa: {kappa:.3f}")

print("\n--- matrice de confusion (lignes=lexic, colonnes=trans)")
print(pd.DataFrame(cm, index=["lexic=0","lexic=1"], columns=["trans=0","trans=1"]))

# mcnemar: 2x2 table (b = lexic=0, trans=1 ; c = lexic=1, trans=0)
b = int(((y_l == 0) & (y_t == 1)).sum())
c = int(((y_l == 1) & (y_t == 0)).sum())
table = [[int((y_l==0).sum() - b), b],
         [c, int((y_l==1).sum() - c)]]
res = mcnemar(table, exact=False, correction=True)
print("\n--- test de mcnemar")
print(f"b (lexic=0, trans=1) = {b}, c (lexic=1, trans=0) = {c}")
print(f"statistique: {res.statistic:.3f}, p-value: {res.pvalue:.4f}")

# 3) prevalence
prev_trans = y_t.mean()
prev_lexic = y_l.mean()
print("\n--- prevalence des positifs")
print(f"trans:  {prev_trans:.3f}")
print(f"lexic:  {prev_lexic:.3f}")
print(f"ecart:  {prev_trans - prev_lexic:+.3f}")

# 4) echantillons de desaccords
df["agree"] = (y_t == y_l).astype(int)

# top desaccords ou trans tres confiant (si score dispo)
if score_trans_col is not None:
    # garder le score associe a la classe predite (on suppose que 'score_trans' est le score de la pred trans)
    df["score_trans"] = df[score_trans_col].astype(float)
    disag = df[df["agree"]==0].copy()
    top_trans_high = disag.sort_values("score_trans", ascending=False).head(20)
    print("\n--- top 5 desaccords (trans tres confiant)")
    print(top_trans_high[["commentaire", y_trans_col, y_lexic_col, "score_trans"]].head(5).to_string(index=False))
else:
    disag = df[df["agree"]==0].copy()
    print("\n--- 5 desaccords aleatoires")
    print(disag.sample(min(5, len(disag)), random_state=42)[["commentaire", y_trans_col, y_lexic_col]].to_string(index=False))

# visualisations
# conf (heatmap simple)
cm_df = pd.DataFrame(cm, index=["lexic=0","lexic=1"], columns=["trans=0","trans=1"])
plt.figure()
plt.imshow(cm_df.values)
plt.xticks(range(2), cm_df.columns)
plt.yticks(range(2), cm_df.index)
for i in range(2):
    for j in range(2):
        plt.text(j, i, cm_df.values[i, j], ha="center", va="center")
plt.title("matrice de confusion (lexic vs trans)")
plt.xlabel("trans")
plt.ylabel("lexic")
plt.tight_layout()
plt.show()

# barres de prevalence
prev_df = pd.DataFrame({
    "methode": ["trans","lexic"],
    "positifs": [prev_trans, prev_lexic]
})
plt.figure()
plt.bar(prev_df["methode"], prev_df["positifs"])
plt.title("prevalence des positifs")
plt.ylabel("part de 1")
plt.tight_layout()
plt.show()

# hist de scores
if score_trans_col is not None:
    plt.figure()
    df["score_trans"].hist(bins=20)
    plt.title("distribution du score trans (tous)")
    plt.xlabel("score_trans")
    plt.ylabel("freq")
    plt.tight_layout()
    plt.show()

    plt.figure()
    df.loc[df["agree"]==1, "score_trans"].hist(bins=20)
    plt.title("score trans sur les accords")
    plt.xlabel("score_trans")
    plt.ylabel("freq")
    plt.tight_layout()
    plt.show()

    plt.figure()
    df.loc[df["agree"]==0, "score_trans"].hist(bins=20)
    plt.title("score trans sur les desaccords")
    plt.xlabel("score_trans")
    plt.ylabel("freq")
    plt.tight_layout()
    plt.show()