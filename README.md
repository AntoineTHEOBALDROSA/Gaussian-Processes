# Processus-Gaussiens — interface locale

Interface en français pour importer un fichier CSV ou Excel, sélectionner la cible et les entrées, lancer une régression par processus gaussien et explorer les graphiques. Les calculs, les fichiers et la bibliothèque de graphiques restent sur la machine. Aucune ressource distante n’est nécessaire après l’installation.

## Démarrer

Depuis le dossier du projet :

```bash
./website/start.sh
```

Ouvrir **http://127.0.0.1:8000**. Garder le terminal ouvert ; `Ctrl+C` arrête le serveur. Pour changer de port : `./website/start.sh --port 8001`.

Le premier lancement installe les dépendances dans `website/.venv` avec `uv` si disponible, sinon avec `python3 -m venv` et pip (Python 3.11 ou supérieur). Cette installation nécessite Internet. Pour réinstaller les dépendances d’un environnement existant :

```bash
uv pip install --python website/.venv/bin/python -r website/requirements.txt
```

Installation manuelle, notamment sous Windows :

```powershell
cd website
py -3.11 -m venv .venv
.venv\Scripts\python -m pip install -r requirements.txt
.venv\Scripts\python app.py
```

## Utilisation

1. Glisser un fichier `.csv` ou `.xlsx`, ou choisir les données `data.xlsx` du projet.
2. Vérifier la cible et les entrées. Les colonnes nommées ID / index ne sont pas sélectionnées par défaut. Les autres identifiants doivent être décochés manuellement.
3. Lancer l’analyse. Le mode complet reprend les paramètres de V1 : 4 noyaux, 5 plis, 2 relances par pli et 10 relances finales. Le mode rapide utilise 3 plis, 0 relance par pli et 1 relance finale.
4. Faire défiler le tableau pour consulter toutes les lignes (affichage virtualisé). L’onglet « Matrice des entrées » est disponible dès l’import et suit les entrées sélectionnées : la ligne donne la variable verticale, la colonne la variable horizontale. La diagonale affiche un histogramme, la moyenne en pointillés et la variance descriptive (division par N). Les valeurs manquantes sont ignorées par variable ou par paire.
5. Après l’analyse, saisir un nouveau point dans « Prédire un nouveau point », puis cliquer sur « Calculer la prédiction ». Le modèle déjà entraîné est réutilisé, sans réapprentissage, avec un intervalle prédictif nominal à 95 %. Les champs correspondent aux entrées sélectionnées, quel que soit leur nom. Une valeur hors des bornes d’apprentissage est signalée comme extrapolation ; être à l’intérieur de ces bornes ne garantit pas la fiabilité.
6. Explorer la parité, les coupes 1D, les données en 1D/2D/3D, la matrice des entrées et le classement des noyaux. La barre d’outils des graphiques permet de zoomer et de télécharger un PNG. Exporter les prédictions de test au format CSV.

Après un entraînement réussi, seuls « Configurer le modèle » et « Explorer les résultats » restent dans la navigation. « Nouveau modèle » remet l’interface à l’import pour choisir un autre jeu de données. Modifier les paramètres conserve le fichier, mais nécessite un nouvel entraînement. Les résultats occupent toute la largeur de la page.

CSV UTF-8 ou Windows-1252, séparateur détecté automatiquement, virgule décimale acceptée avec un séparateur approprié (par exemple `;`). Pour Excel, seule la première feuille est lue. Première ligne : noms de colonnes. Limites : 20 Mo, 3 000 lignes, 50 colonnes, 20 entrées sélectionnées. Le calcul GP exact peut être long sur les fichiers les plus volumineux. Les lignes manquantes, non numériques ou infinies sont exclues uniquement sur les variables sélectionnées ; le nombre est indiqué après analyse. Minimum : 10 lignes valides et 8 configurations distinctes.

Les fichiers, modèles entraînés et résultats sont conservés en mémoire uniquement, jusqu’au redémarrage du serveur (10 imports et 10 analyses au maximum). Un rafraîchissement du même onglet retrouve sa dernière analyse tant que le serveur la conserve. Un seul calcul s’exécute à la fois. Les CSV exportés restent là où le navigateur les télécharge.

## Modèle et provenance

- `models.py` reprend `GPRWrapper` et `noyaux_candidats` de `../V1.ipynb`, avec un choix explicite CPU / CUDA.
- `gpu_model.py` reprend le régresseur PyTorch de `../V1_GPU.ipynb`, chargé seulement en mode CUDA.
- `analysis.py` réalise le découpage, la validation croisée, la sélection, l’ajustement final et la préparation des résultats.
- `api.py` gère l’import, les tâches en arrière-plan, la prédiction de nouveaux points et les exports.
- `static/exploration.js` gère le tableau virtualisé, la matrice des entrées et le formulaire de prédiction.
- `app.py` sert l’interface sur l’adresse de boucle locale uniquement.

Les notebooks et `main.py` restent inchangés. Les modules sont des copies autonomes : une modification future du notebook doit être reportée explicitement.

La graine est 2025. Les statistiques de normalisation sont ajustées sur l’apprentissage uniquement, dans chaque pli. Le partage 80/20 est effectué par configurations distinctes : les doublons restent ensemble en apprentissage/test, puis dans chaque pli de validation croisée. Ce partage diffère du partage par lignes du notebook pour éviter la fuite de données ; les métriques peuvent donc différer. Pour les coupes, les entrées non tracées sont fixées à leur médiane d’apprentissage. Les intervalles sont des intervalles prédictifs nominaux à 95 % du modèle, sans garantie de couverture réelle.

## Modifier le calcul Python

Le site n’exécute pas les fichiers `.ipynb`. Il importe directement les modules Python de `website` ; modifier un notebook ne change donc pas le site automatiquement.

| Fichier | Rôle / modifications |
| --- | --- |
| `models.py` → `noyaux_candidats(n_dim)` | Liste des noyaux, valeurs initiales et bornes de leurs hyperparamètres. Supprimer une entrée du dictionnaire `signaux` retire un candidat ; en ajouter une ajoute un candidat. Garder au moins un candidat. |
| `models.py` → `GPRWrapper` | Normalisation des entrées / sorties, choix CPU / CUDA, ajustement et prédiction dans les unités originales. |
| `analysis.py` → `prepare_data` / `run_analysis` | Partage apprentissage-test, nombre de plis, relances d’optimisation, métriques et données des graphiques. |
| `gpu_model.py` | Calcul matriciel PyTorch du notebook GPU. À adapter si un nouveau type de noyau doit fonctionner sur CUDA. |
| `api.py` | Import des fichiers, lancement des analyses, conservation des modèles et prédiction des nouveaux points. |

Exemple : dans `noyaux_candidats`, remplacer la ligne du RBF par ceci pour rechercher chaque longueur de corrélation entre 0,1 et 100 :

```python
"RBF (ARD)": RBF(
    length_scale=np.ones(n_dim),
    length_scale_bounds=(1e-1, 1e2),
),
```

Pour les Matérn, modifier leur deuxième argument `(1e-2, 1e3)`, ou utiliser également le mot-clé `length_scale_bounds`. Une paire de bornes s’applique à tous les ℓᵢ ; une matrice `(n_dim, 2)` permet des bornes différentes pour chaque entrée, dans l’ordre des variables sélectionnées sur le site. Choisir des bornes strictement positives et une valeur initiale à l’intérieur.

Les entrées sont standardisées avant l’ajustement : ces ℓᵢ sont exprimés dans l’espace normalisé, pas dans les unités originales des Xᵢ. Dans ce code, RBF et Matérn sont anisotropes (une longueur par entrée), tandis que RationalQuadratic utilise une seule longueur commune. `ConstantKernel` règle l’amplitude et `WhiteKernel` le bruit ; leurs valeurs initiales et bornes sont à la fin de `noyaux_candidats`.

Après une modification Python : arrêter le serveur avec `Ctrl+C`, puis relancer `./website/start.sh` depuis la racine du projet. Actualiser la page, réimporter les données et réentraîner le modèle. Le serveur n’active pas le rechargement automatique ; ses anciens modèles en mémoire ne changent pas quand un fichier est modifié. Les modifications HTML/CSS/JS ne nécessitent qu’une actualisation du navigateur.

Le CPU utilise les noyaux scikit-learn. Le backend CUDA actuel accepte uniquement `ConstantKernel * (RBF / Matérn ν=1,5 ou 2,5 / RationalQuadratic) + WhiteKernel` : l’ajout d’un autre noyau demande aussi une implémentation dans `gpu_model.py`, ou l’utilisation du CPU.

## GPU optionnel

Le CPU fonctionne sans PyTorch. Pour CUDA, installer une distribution PyTorch adaptée à la machine dans `website/.venv`, puis redémarrer le serveur. Le choix GPU s’active seulement si CUDA est accessible et la factorisation float64 fonctionne. Aucun remplacement silencieux d’un GPU demandé par le CPU. La voie GPU n’a pas été validée sur une carte CUDA dans cet environnement.

## Vérifications

```bash
cd website
.venv/bin/python -m unittest discover -s tests -v
```

Ce serveur est destiné à l’usage local sur une seule machine, avec stockage temporaire en mémoire.
# Gaussian-Processes
