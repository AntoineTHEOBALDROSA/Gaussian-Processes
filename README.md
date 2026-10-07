# Gaussian-Processes

## Démarrer

Depuis la racine du projet :

```bash
./website/start.sh
```

Ouvrir **http://127.0.0.1:8000**.

## Dépendances

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

Le bouton **Comment ça fonctionne ?**, en haut de l’interface, ouvre une page d’explication dans un nouvel onglet. Son schéma illustre le partage apprentissage/test, la boucle de validation croisée, le choix du noyau et l’ajustement final. Les modes rapide/complet et les banques de 4/8 noyaux peuvent être illustrés sans modifier l’analyse. La page détaille aussi les paramètres, les prédictions Excel, la sauvegarde et les propositions de simulations. Une galerie de huit noyaux montre leurs formules et leurs prédictions sur les mêmes observations 1D synthétiques, avec des paramètres fixes. Les graphiques SVG locaux peuvent être régénérés avec `tools/generate_kernel_examples.py` (scikit-learn pour le calcul, Matplotlib pour le rendu). Elle est servie à `/static/fonctionnement.html` et fonctionne entièrement en local.

1. Glisser un fichier `.csv` ou `.xlsx`, ou choisir les données `data.xlsx` du projet.
2. Vérifier la cible et les entrées. Les colonnes nommées ID / index ne sont pas sélectionnées par défaut. Les autres identifiants doivent être décochés manuellement.
3. Lancer l’analyse. La banque simple reprend les 4 noyaux de V1. Les réglages proposent aussi une banque élargie de 8 noyaux sur CPU et des bornes de longueurs personnalisées. Le mode complet utilise 5 plis, 2 relances par pli et 10 relances finales ; le mode rapide utilise 3 plis, 0 relance par pli et 1 relance finale.
4. Faire défiler le tableau pour consulter toutes les lignes (affichage virtualisé). L’onglet « Matrice des entrées » est disponible dès l’import et suit les entrées sélectionnées : la ligne donne la variable verticale, la colonne la variable horizontale. La diagonale affiche un histogramme, la moyenne en pointillés et la variance descriptive (division par N). Les valeurs manquantes sont ignorées par variable ou par paire.
5. Après l’analyse, saisir un nouveau point dans « Prédire un nouveau point », puis cliquer sur « Calculer la prédiction ». Le modèle déjà entraîné est réutilisé, sans réapprentissage, avec un intervalle prédictif nominal à 95 %. Les champs correspondent aux entrées sélectionnées, quel que soit leur nom. Une valeur hors des bornes d’apprentissage est signalée comme extrapolation ; être à l’intérieur de ces bornes ne garantit pas la fiabilité.
6. Explorer la parité, les coupes 1D, les données en 1D/2D/3D, la matrice des entrées et le classement des noyaux. La barre d’outils des graphiques permet de zoomer et de télécharger un PNG. Exporter les prédictions de test au format CSV.

Après un entraînement réussi, seuls « Configurer le modèle » et « Explorer les résultats » restent dans la navigation. « Nouveau modèle » remet l’interface à l’import pour choisir un autre jeu de données. Modifier les paramètres conserve le fichier, mais nécessite un nouvel entraînement. Les résultats occupent toute la largeur de la page.

CSV UTF-8 ou Windows-1252, séparateur détecté automatiquement, virgule décimale acceptée avec un séparateur approprié (par exemple `;`). Pour l’entraînement depuis Excel, seule la première feuille est lue. Première ligne : noms de colonnes. Limites : 20 Mo, 3 000 lignes, 50 colonnes, 20 entrées sélectionnées. Le calcul GP exact peut être long sur les fichiers les plus volumineux. Les lignes manquantes, non numériques ou infinies sont exclues uniquement sur les variables sélectionnées ; le nombre est indiqué après analyse. Minimum : 10 lignes valides et 8 configurations distinctes.

Les études ouvertes sont conservées en mémoire jusqu’au redémarrage du serveur (10 imports et 10 analyses au maximum). Pour les conserver durablement, télécharger une sauvegarde `.gpmodel` avec « Sauvegarder le modèle ». Un rafraîchissement du même onglet retrouve sa dernière analyse tant que le serveur la conserve. Un seul calcul s’exécute à la fois. Les CSV exportés restent là où le navigateur les télécharge.

## Sauvegarder et rouvrir un modèle

Après l’entraînement, cliquer sur **Sauvegarder le modèle**. Le fichier `.gpmodel` téléchargé contient les paramètres appris, les normalisations, la factorisation et les coefficients du GP, le jeu de données importé, les réglages, les résultats des graphiques et les prédictions de test.

Cliquer sur **Ouvrir un modèle**, disponible dès l’accueil, puis sélectionner cette sauvegarde (100 Mo maximum). Les résultats et la prédiction de nouveaux points sont immédiatement disponibles, même après un redémarrage du serveur. Aucun entraînement ni optimisation n’est relancé. La réouverture calcule les nouvelles prédictions sur CPU, y compris pour un modèle initialement entraîné sur CUDA ; ce choix est indiqué dans les résultats. Le moteur d’entraînement initial reste indiqué dans le bilan. Un réentraînement éventuel se règle depuis « Configurer le modèle ».

Le format versionné contient du JSON et des tableaux NumPy numériques dans une archive ZIP : aucun pickle et aucun code chargé depuis le fichier. Les fichiers incomplets ou incompatibles sont rejetés. La sauvegarde prend en charge les noyaux actuels et leurs sommes/produits (ConstantKernel, WhiteKernel, RBF, Matérn 1/2, 3/2 ou 5/2, RationalQuadratic, DotProduct et puissances 1 à 3). Les anciennes sauvegardes restent compatibles. L’ajout d’un autre noyau nécessite d’étendre `model_io.py` pour sa sauvegarde.

## Prédire dans un classeur Excel

Après entraînement ou réouverture, ouvrir **Prédire depuis un fichier Excel** dans les résultats. Choisir un `.xlsx`, la feuille et la ligne des titres, puis cliquer sur **Lire les colonnes** si ces réglages changent. Associer chaque entrée du modèle à sa colonne Excel ; les noms identiques sont présélectionnés. Les colonnes sont identifiées par leurs lettres et acceptent des titres vides ou répétés.

Choisir la colonne de sortie existante, par exemple `G — FprédiX`, ou ajouter une nouvelle colonne avec son nom. Définir la première et la dernière ligne à traiter puis **Prédire et télécharger le classeur**. Le fichier d’origine reste intact : le navigateur télécharge une copie `…-predictions.xlsx` contenant les valeurs numériques prédites.

Par défaut, les cellules de sortie déjà remplies, y compris celles contenant une formule, sont conservées. Cocher **Remplacer les valeurs déjà présentes** pour les remplacer par les prédictions. Les entrées vides, non numériques ou infinies font ignorer la ligne ; le bilan indique leur nombre et les 20 premiers numéros concernés. Les chaînes à virgule décimale sont acceptées. Une formule utilisée en entrée doit avoir une valeur calculée enregistrée par Excel ; le serveur ne recalcule pas les formules. Les cellules fusionnées dans la zone de sortie sont refusées.

Limites distinctes de l’entraînement : 20 Mo, 200 colonnes par feuille, 10 000 lignes par traitement. Aucun réentraînement. Les cellules de sortie choisies sont modifiées directement dans le XML du classeur ; les autres parties sont recopiées, conservant notamment les styles, commentaires, dessins, autres feuilles et formules. Le classeur demande à Excel de recalculer les formules à son ouverture. Un nouveau champ ajouté n’étend pas automatiquement une table structurée Excel existante.

## Comparer les noyaux et interpréter les bornes

Dans **Configurer le modèle → Réglages de l’analyse**, la banque élargie ajoute : noyau linéaire, quadratique, tendance linéaire + Matérn 3/2, et somme de deux RBF ARD pour des échelles différentes. Elle nécessite le CPU. Les 8 candidats utilisent les mêmes plis de validation par groupes ; le classement se fait par RMSE moyenne, jamais par le test final. Davantage de candidats ne garantit pas une meilleure généralisation. Pour comparer de nombreuses variantes et bornes, une validation imbriquée ou un nouveau test indépendant serait préférable.

Les bornes réglables concernent les longueurs de corrélation, dans l’espace standardisé. Les bornes d’amplitude, bruit et autres paramètres restent dans `models.py`. L’onglet **Noyaux** donne le diagnostic des paramètres du modèle final : une valeur située à moins de 0,1 % de l’intervalle en échelle logarithmique est signalée. Un bruit proche de zéro peut être normal pour des simulations déterministes ; une grande longueur peut indiquer une variation lente ou une variable peu influente. Modifier les bornes est une hypothèse à tester par validation, sans ajuster les choix pour améliorer le score du test final.

## Identifier des configurations à mesurer

L’onglet **Données à compléter** affiche la distribution des sorties réelles (12 classes de largeur égale) et les trois plages les moins représentées. Une sortie rare n’est pas, à elle seule, une preuve de mauvaise couverture du domaine des entrées.

Indiquer les bornes physiques des entrées, et cocher **Entiers** pour des variables discrètes. Les bornes observées servent de point de départ ; toutes les valeurs entières à l’import entraînent une présélection à vérifier. Demander 1 à 30 simulations. On peut cibler une plage de sortie **prédite**, par exemple F entre 2 et 3,5 ; cela ne garantit pas que la mesure future sera dans cette plage.

La recherche génère 2 048 candidats par hypercube latin (graine 2025), arrondit les entrées discrètes, élimine les doublons et les configurations déjà connues, puis calcule leurs prédictions. Le score combine à parts égales les rangs de distance au plus proche voisin et d’incertitude de la fonction GP. La distance standardise chaque entrée avec l’ensemble des configurations connues. L’incertitude soustrait la variance de bruit du noyau : un bruit d’observation élevé ne doit pas, à lui seul, appeler davantage de mesures. Une sélection successive favorise ensuite la diversité du lot.

Le tableau contient les entrées, la sortie estimée, σ de la fonction et la distance aux observations ; le graphe projette les points sur deux entrées au choix. **Exporter les configurations (.csv)** prépare les entrées et une colonne cible vide pour les résultats des futures simulations. Les propositions sont exploratoires : l’incertitude dépend du noyau et n’est pas calibrée automatiquement ; les projections 2D peuvent masquer des trous en dimensions supérieures ; le nombre de candidats est limité. Les bornes définissent une boîte sans contraintes entre les variables. La faisabilité physique doit être validée avant de demander les simulations à Dassault. Aucun résultat prédit n’est ajouté aux observations d’apprentissage.

**Exporter les graphiques en PDF** produit l’histogramme des sorties et, si un lot a été proposé, sa projection sur les axes actuellement sélectionnés. Utiliser le CSV pour obtenir les valeurs des configurations elles-mêmes.

## Exporter les graphiques en PDF

Dans les résultats, cliquer sur **Exporter en PDF** pour télécharger la vue choisie : prédictions, coupe 1D de la variable sélectionnée, données, matrice des entrées, ou comparaison des noyaux (représentée en barres avec l’écart-type de la RMSE entre plis). La vue 3D conserve son orientation actuelle. Le zoom du graphique 2D est conservé. La matrice est également exportable avant entraînement depuis son onglet.

Les PDF utilisent des pages A4 paysage avec titre et description. Les graphiques sont intégrés en images haute résolution ; ils ne sont pas vectoriels. Chaque matrice est répartie à raison de quatre cases par page, y compris les cases pas encore affichées à l’écran. La moyenne et la variance des diagonales figurent dans l’export. Le traitement et l’export restent entièrement locaux et ne nécessitent aucun service externe, navigateur supplémentaire ou installation de Chrome. Les exports PNG et CSV restent disponibles.

## Modèle et provenance

- `models.py` reprend `GPRWrapper` et `noyaux_candidats` de `../V1.ipynb`, avec un choix explicite CPU / CUDA.
- `gpu_model.py` reprend le régresseur PyTorch de `../V1_GPU.ipynb`, chargé seulement en mode CUDA.
- `analysis.py` réalise le découpage, la validation croisée, la sélection, l’ajustement final et la préparation des résultats.
- `api.py` gère l’import, les tâches en arrière-plan, la prédiction de nouveaux points et les exports.
- `model_io.py` sauvegarde et rouvre les études et leur modèle portable.
- `pdf_export.py` assemble les graphiques haute résolution en PDF avec ReportLab.
- `excel_prediction.py` lit les correspondances de colonnes et complète une copie du classeur sans réenregistrer ses autres parties.
- `acquisition.py` propose des configurations complémentaires selon distance, incertitude et diversité.
- `static/research.js` gère les prédictions Excel, les diagnostics des bornes et l’exploration des données à compléter.
- `static/transfers.js` gère la sauvegarde, la réouverture et les captures PDF côté navigateur.
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

Ce serveur est destiné à l’usage local sur une seule machine. Les études ouvertes restent en mémoire ; les sauvegardes et exports sont conservés dans les fichiers téléchargés.
# Gaussian-Processes
