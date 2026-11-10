"""Génère et exécute les notebooks (sorties enregistrées).

Usage : python scripts/build_notebooks.py
"""
from pathlib import Path

import nbformat as nbf
from nbclient import NotebookClient

ROOT = Path(__file__).resolve().parents[1]
NB = ROOT / "notebooks"
SETUP = """import sys
from pathlib import Path
ROOT = Path.cwd().resolve().parent if Path.cwd().name == "notebooks" else Path.cwd().resolve()
sys.path.insert(0, str(ROOT))
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from src.data import download, load_ratings, load_movies, GENRES
download()
VIOLET, GOLD = "#8B5CF6", "#F5C451"
"""

EXPLORATION = [
    ("md", "# 01 — Exploration de MovieLens 100K\n\nPremière prise en main du jeu de données : volumes, distribution des notes, genres, films les mieux notés et activité des utilisateurs. Données : GroupLens Research (voir le README pour les conditions d'utilisation)."),
    ("code", SETUP),
    ("code", "ratings = load_ratings()\nmovies = load_movies()\nprint(f\"{len(ratings):,} notes · {ratings.user_id.nunique()} utilisateurs · {ratings.item_id.nunique()} films\")\nprint('Valeurs manquantes :', int(ratings.isna().sum().sum()))\nprint('Doublons utilisateur–film :', int(ratings.duplicated(['user_id', 'item_id']).sum()))\ndensite = len(ratings) / (ratings.user_id.nunique() * ratings.item_id.nunique())\nprint(f'Densité de la matrice : {densite:.2%}')\nratings.head()"),
    ("md", "## Distribution des notes"),
    ("code", "dist = ratings.rating.value_counts().sort_index()\nax = dist.plot.bar(color=[VIOLET, VIOLET, VIOLET, GOLD, VIOLET], rot=0, figsize=(6, 3.5))\nax.set_xlabel('Note'); ax.set_ylabel('Nombre de notes'); ax.set_title('Distribution des 100 000 notes')\nplt.show()\n(dist / len(ratings) * 100).round(2).rename('%').to_frame().assign(notes=dist)"),
    ("md", "## Notes par genre\n\nUn film peut appartenir à plusieurs genres : chaque note est alors comptée dans chacun de ses genres (le total dépasse donc 100 000)."),
    ("code", "data = ratings.merge(movies, on='item_id')\ngenres = pd.Series({g: int(data[g].sum()) for g in GENRES}).sort_values(ascending=False)\ngenres.head(8).sort_values().plot.barh(color=VIOLET, figsize=(6, 3.5), title='8 genres les plus notés')\nplt.show()\ngenres.head(8)"),
    ("md", "## Films les mieux notés (au moins 100 notes)\n\nSans seuil, un film noté une seule fois 5/5 arriverait en tête : le seuil rend la moyenne plus fiable."),
    ("code", "stats = data.groupby(['item_id', 'title']).rating.agg(moyenne='mean', notes='count').reset_index()\neligibles = stats[stats.notes >= 100]\nprint(f'{len(eligibles)} films ont au moins 100 notes')\neligibles.sort_values(['moyenne', 'notes'], ascending=False).head(5).round(3)"),
    ("md", "## Activité des utilisateurs\n\nChaque utilisateur a noté au moins 20 films, mais l'activité est très inégale : une longue traîne que les modèles doivent gérer."),
    ("code", "par_user = ratings.groupby('user_id').size()\nax = par_user.plot.hist(bins=range(0, 801, 50), color=VIOLET, figsize=(6, 3.5))\nax.axvline(par_user.median(), color=GOLD, lw=3, label=f'médiane = {par_user.median():.0f}')\nax.set_xlabel('Notes par utilisateur'); ax.set_ylabel('Utilisateurs'); ax.legend()\nplt.show()\npar_user.describe().round(1)"),
    ("md", "## À retenir\n\n- Matrice très creuse (~6 % de cases remplies) : c'est exactement le problème que la recommandation cherche à combler.\n- Les notes sont plutôt positives (4 est la plus fréquente) : on considérera un film comme « pertinent » à partir de 4.\n- Popularité et activité suivent une longue traîne : une référence par popularité sera difficile à battre en top-N."),
]

MODELES = [
    ("md", "# 02 — Modèles de recommandation\n\nTrois approches comparées sur le découpage officiel **u1.base / u1.test** (80 000 / 20 000 notes) :\n\n1. **Popularité / moyenne** : note = moyenne lissée du film ; top-N = films les plus notés.\n2. **Item-KNN** : similarité cosinus ajustée entre films, 40 voisins.\n3. **Factorisation matricielle** biaisée (type Funk SVD), entraînée par SGD.\n\nMétriques : RMSE et MAE sur les notes, Precision@10 et Recall@10 (pertinent = note ≥ 4 dans le test, films vus à l'entraînement exclus)."),
    ("code", SETUP + "from src.data import load_split\nfrom src.models import PopularityBaseline, ItemKNN, MatrixFactorization\nfrom src.metrics import rmse, mae, evaluate_top_n\ntrain, test = load_split(1)\nprint(len(train), 'notes d\\'entraînement ·', len(test), 'notes de test')"),
    ("code", "models = {\n    'Popularité / moyenne': PopularityBaseline(damping=5),\n    'Item-KNN (cosinus)': ItemKNN(k=40, shrinkage=10),\n    'Factorisation (SVD-SGD)': MatrixFactorization(n_factors=50, n_epochs=20, lr=0.01, reg=0.05, seed=42),\n}\nrows = []\nfor name, model in models.items():\n    model.fit(train, valid=test) if isinstance(model, MatrixFactorization) else model.fit(train)\n    pred = model.predict(test.user_id, test.item_id)\n    top = evaluate_top_n(model, test, k=10)\n    rows.append({'modèle': name, 'RMSE': rmse(test.rating, pred), 'MAE': mae(test.rating, pred),\n                 'Precision@10': top['precision@10'], 'Recall@10': top['recall@10']})\nresults = pd.DataFrame(rows).set_index('modèle').round(4)\nresults"),
    ("md", "## Courbe d'apprentissage de la factorisation"),
    ("code", "hist = pd.DataFrame(models['Factorisation (SVD-SGD)'].history_)\nax = hist.plot(x='epoch', y=['train_rmse', 'test_rmse'], marker='o', color=[VIOLET, GOLD], figsize=(6, 3.5))\nax.set_xlabel('Époque'); ax.set_ylabel('RMSE'); ax.set_title('SGD : RMSE par époque')\nplt.show()\nhist.tail(3).round(4)"),
    ("md", "## Exemple : recommandations pour l'utilisateur 1"),
    ("code", "movies = load_movies().set_index('item_id').title\nvus = train[train.user_id == 1].sort_values('rating', ascending=False).head(5)\nprint('Quelques films qu\\'il a adorés :')\nprint(vus.item_id.map(movies).to_string(index=False))\npd.DataFrame({name: [movies[i] for i in m.recommend(1, 10)] for name, m in models.items()}, index=range(1, 11))"),
    ("md", "## Lecture des résultats\n\n- La **factorisation** prédit le mieux les notes (RMSE la plus basse).\n- **Item-KNN** fournit le meilleur top-10 : prédire une note précise et classer les films sont deux problèmes différents.\n- Classer par note prédite favorise des films peu vus avec une forte moyenne estimée, d'où une Precision@10 plus faible pour la factorisation ; la **popularité** reste une référence solide.\n- Pistes : optimiser directement le classement (BPR, ALS implicite), validation croisée sur les 5 découpages, hyperparamètres."),
]


def build(name, cells):
    nb = nbf.v4.new_notebook()
    nb.cells = [nbf.v4.new_markdown_cell(s) if kind == "md" else nbf.v4.new_code_cell(s) for kind, s in cells]
    nb.metadata["kernelspec"] = {"name": "python3", "display_name": "Python 3", "language": "python"}
    NotebookClient(nb, timeout=900, kernel_name="python3", resources={"metadata": {"path": str(NB)}}).execute()
    nbf.write(nb, NB / name)
    print("exécuté :", NB / name)


if __name__ == "__main__":
    build("01_exploration.ipynb", EXPLORATION)
    build("02_modeles.ipynb", MODELES)
