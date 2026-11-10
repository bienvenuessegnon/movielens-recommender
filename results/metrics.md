# Résultats — découpage officiel u1.base (80000 notes) / u1.test (20000 notes)

| modele | rmse | mae | precision@10 | recall@10 | utilisateurs_top_n |
| --- | --- | --- | --- | --- | --- |
| Popularité / moyenne | 1.0370 | 0.8319 | 0.2129 | 0.1179 | 456 |
| Item-KNN (cosinus) | 0.9963 | 0.7622 | 0.2947 | 0.1654 | 456 |
| Factorisation (SVD-SGD) | 0.9339 | 0.7353 | 0.1333 | 0.0479 | 456 |

RMSE/MAE : erreur sur les notes du test (plus bas = mieux).
Precision@10 / Recall@10 : un film est pertinent s'il est noté ≥ 4 dans le test ; les films déjà vus à l'entraînement sont exclus des recommandations.
