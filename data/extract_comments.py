import json, csv, sys
from pathlib import Path

info = json.load(open("data\VKancdDIpOU.info.json", "r", encoding="utf-8"))
comments = info.get("comments", [])
with open("data\comments.csv", "w", newline="", encoding="utf-8-sig") as f:
    w = csv.DictWriter(f, fieldnames=["id","author","text","like_count","timestamp","parent"])
    w.writeheader()
    for c in comments:
        w.writerow({
            "id": c.get("id"),
            "author": c.get("author"),
            "text": c.get("text"),
            "like_count": c.get("like_count"),
            "timestamp": c.get("timestamp"),
            "parent": c.get("parent"),
        })
print("OK → data\comments.csv")

# create a csv with one comment per line
with open("data\comments_only.csv", "w", newline="", encoding="utf-8-sig") as f:
    w = csv.writer(f)
    w.writerow(["text"])
    for c in comments:
        autor = c.get("author", "").replace("\n", " ").replace("\r", " ").strip()
        text = c.get("text", "").replace("\n", " ").replace("\r", " ").strip()
        comment = f"{autor} {text}".strip()
        if comment:
            w.writerow([comment])

print("OK → data\comments_only.csv")