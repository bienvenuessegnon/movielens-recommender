"""Entraîne les trois modèles sur u1.base, évalue sur u1.test, écrit results/.

Usage : python -m src.evaluate
"""
from __future__ import annotations

import argparse
import json
import time
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import pandas as pd

from .data import ROOT, load_split
from .metrics import evaluate_top_n, mae, rmse
from .models import ItemKNN, MatrixFactorization, PopularityBaseline

RESULTS = ROOT / "results"
VIOLET, GOLD, INK = "#8B5CF6", "#F5C451", "#1A1A2E"
LABELS = {
    "popularite": "Popularité / moyenne",
    "itemknn": "Item-KNN (cosinus)",
    "svd": "Factorisation (SVD-SGD)",
}


def build_models() -> dict:
    return {
        "popularite": PopularityBaseline(damping=5),
        "itemknn": ItemKNN(k=40, shrinkage=10),
        "svd": MatrixFactorization(n_factors=50, n_epochs=20, lr=0.01, reg=0.05, seed=42),
    }


def run(fold: int = 1, k: int = 10, out: Path = RESULTS) -> pd.DataFrame:
    train, test = load_split(fold)
    out.mkdir(parents=True, exist_ok=True)
    rows, history = [], None
    for name, model in build_models().items():
        start = time.perf_counter()
        if name == "svd":
            model.fit(train, valid=test)
            history = pd.DataFrame(model.history_)
        else:
            model.fit(train)
        fit_s = time.perf_counter() - start
        pred = model.predict(test["user_id"], test["item_id"])
        top = evaluate_top_n(model, test, k=k)
        rows.append({
            "modele": LABELS[name],
            "rmse": rmse(test["rating"], pred),
            "mae": mae(test["rating"], pred),
            f"precision@{k}": top[f"precision@{k}"],
            f"recall@{k}": top[f"recall@{k}"],
            "utilisateurs_top_n": top["users_evaluated"],
            "entrainement_s": fit_s,
        })
        print(f"{LABELS[name]:<26} RMSE={rows[-1]['rmse']:.4f} MAE={rows[-1]['mae']:.4f} "
              f"P@{k}={rows[-1][f'precision@{k}']:.4f} R@{k}={rows[-1][f'recall@{k}']:.4f}")
    table = pd.DataFrame(rows)
    table.to_csv(out / "metrics.csv", index=False, float_format="%.4f")
    history.to_csv(out / "svd_training.csv", index=False, float_format="%.4f")
    _markdown(table, out / "metrics.md", fold, k, len(train), len(test))
    _charts(table, history, out, k)
    (out / "run_info.json").write_text(json.dumps({
        "split": f"u{fold}.base / u{fold}.test (officiel GroupLens)",
        "train_ratings": len(train), "test_ratings": len(test),
        "relevance_threshold": 4, "k": k,
    }, indent=2))
    return table


def _markdown(table, path, fold, k, n_train, n_test):
    shown = table.drop(columns=["entrainement_s"])
    header = "| " + " | ".join(shown.columns) + " |"
    sep = "| " + " | ".join(["---"] * len(shown.columns)) + " |"
    lines = [header, sep]
    for _, row in shown.iterrows():
        cells = [f"{v:.4f}" if isinstance(v, float) else str(v) for v in row]
        lines.append("| " + " | ".join(cells) + " |")
    path.write_text(
        f"# Résultats — découpage officiel u{fold}.base ({n_train} notes) / u{fold}.test ({n_test} notes)\n\n"
        + "\n".join(lines)
        + f"\n\nRMSE/MAE : erreur sur les notes du test (plus bas = mieux).\n"
        f"Precision@{k} / Recall@{k} : un film est pertinent s'il est noté ≥ 4 dans le test ; "
        "les films déjà vus à l'entraînement sont exclus des recommandations.\n"
    )


def _charts(table, history, out, k):
    plt.rcParams.update({"font.size": 11, "axes.spines.top": False, "axes.spines.right": False})
    names = table["modele"].tolist()

    fig, ax = plt.subplots(figsize=(8, 4.5))
    x = range(len(names))
    ax.bar([i - 0.2 for i in x], table["rmse"], 0.4, label="RMSE", color=VIOLET)
    ax.bar([i + 0.2 for i in x], table["mae"], 0.4, label="MAE", color=GOLD)
    for i, (r, m) in enumerate(zip(table["rmse"], table["mae"])):
        ax.text(i - 0.2, r + 0.01, f"{r:.3f}", ha="center", fontsize=9)
        ax.text(i + 0.2, m + 0.01, f"{m:.3f}", ha="center", fontsize=9)
    ax.set_xticks(list(x), names)
    ax.set_ylabel("Erreur (étoiles)")
    ax.set_title("Erreur de prédiction des notes — u1.test (plus bas = mieux)")
    ax.legend(frameon=False)
    fig.tight_layout(); fig.savefig(out / "erreurs_notes.png", dpi=150); plt.close(fig)

    fig, ax = plt.subplots(figsize=(8, 4.5))
    ax.bar([i - 0.2 for i in x], table[f"precision@{k}"], 0.4, label=f"Precision@{k}", color=VIOLET)
    ax.bar([i + 0.2 for i in x], table[f"recall@{k}"], 0.4, label=f"Recall@{k}", color=GOLD)
    for i, (p, r) in enumerate(zip(table[f"precision@{k}"], table[f"recall@{k}"])):
        ax.text(i - 0.2, p + 0.003, f"{p:.3f}", ha="center", fontsize=9)
        ax.text(i + 0.2, r + 0.003, f"{r:.3f}", ha="center", fontsize=9)
    ax.set_xticks(list(x), names)
    ax.set_title(f"Qualité du top-{k} — pertinent = note ≥ 4 dans u1.test")
    ax.legend(frameon=False)
    fig.tight_layout(); fig.savefig(out / "top_n.png", dpi=150); plt.close(fig)

    fig, ax = plt.subplots(figsize=(8, 4.5))
    ax.plot(history["epoch"], history["train_rmse"], marker="o", color=VIOLET, label="entraînement")
    ax.plot(history["epoch"], history["test_rmse"], marker="o", color=GOLD, label="test (u1.test)")
    ax.set_xlabel("Époque"); ax.set_ylabel("RMSE")
    ax.set_title("Factorisation matricielle : apprentissage par SGD")
    ax.legend(frameon=False)
    fig.tight_layout(); fig.savefig(out / "svd_apprentissage.png", dpi=150); plt.close(fig)


def main() -> None:
    parser = argparse.ArgumentParser(description="Évalue les modèles de recommandation")
    parser.add_argument("--fold", type=int, default=1)
    parser.add_argument("--k", type=int, default=10)
    args = parser.parse_args()
    run(args.fold, args.k)


if __name__ == "__main__":
    main()
