# Processus-Gaussiens

## Démarrer

Depuis la racine du projet :

```bash
./website/start.sh
```

Ouvrir **http://127.0.0.1:8000**. Le serveur écoute seulement sur la boucle locale.
Le premier démarrage crée `website/.venv` et installe `requirements.txt` (Internet nécessaire à l’installation uniquement).
Pour mettre les dépendances à jour :

```bash
uv pip install --python website/.venv/bin/python -r website/requirements.txt
```

Sans uv, utiliser `website/.venv/bin/python -m pip install -r website/requirements.txt`.
Sous Windows : créer un environnement Python 3.11 ou supérieur, installer les mêmes dépendances, puis lancer `.venv\Scripts\python app.py` dans `website`.

## Choisir les variables et les sorties

Importer un CSV ou un XLSX (première feuille, première ligne = titres). Limites : 20 Mo et 3 000 observations. Il n’y a plus de limite fixe à 3 ou 20 entrées ni à 50 colonnes : toutes les colonnes importées peuvent être sélectionnées. La mémoire et le temps du GP exact restent des limites pratiques. L’encodage dense des entrées est limité à 256 Mo.

Chaque ligne de la liste de variables contient un rôle **Ignorer / Entrée / Sortie** et un type **Numérique / Catégorie**. La recherche et les commandes de sélection des entrées facilitent les fichiers larges. Une colonne ne peut pas être simultanément entrée et sortie. Les identifiants reconnus sont ignorés par défaut.

On peut sélectionner plusieurs sorties et mélanger régression et classification. Les catégories codées par des nombres doivent être explicitement déclarées « Catégorie ». Les types automatiques sont des suggestions. Les lignes incomplètes pour l’ensemble des variables choisies sont exclues et comptées. Minimum : 10 lignes complètes et 8 configurations distinctes. Une classe doit être suffisamment représentée pour être apprise dans les plis.

Un modèle indépendant est entraîné pour chaque sortie. Les corrélations entre sorties ne sont pas modélisées. Les entrées catégorielles sont encodées en indicatrices (one-hot) apprises sur l’apprentissage de chaque pli, puis les colonnes sont standardisées. Chaque sortie numérique est normalisée séparément. Une classe de sortie est traitée par un **GaussianProcessClassifier**, approximation de Laplace et intégration logistique par quadrature de Gauss-Hermite à 32 points. En multiclasse : un contre tous, probabilités normalisées. Une catégorie inconnue dans un pli/test est encodée par un vecteur nul ; les modalités inédites dans le test sont signalées. Une nouvelle prédiction refuse les modalités inconnues de son apprentissage.

## Entraîner et explorer

Le partage par configurations est de 80 % pour l’apprentissage et 20 % pour le test. Les doublons restent ensemble, y compris dans la validation croisée. La graine de départ est 2025. Si nécessaire, le partage est adapté pour conserver les classes dans les jeux d’apprentissage ; la graine retenue est affichée. Les statistiques d’encodage et de normalisation sont apprises dans chaque pli, sans utiliser le test.

Le mode rapide utilise 3 plis, 0 relance par pli et 1 relance finale. Le mode complet utilise 5 plis, 2 relances par pli et 10 relances finales. La banque simple contient RBF ARD, Matérn 3/2, Matérn 5/2 et RationalQuadratic. La banque élargie ajoute linéaire, quadratique, linéaire + Matérn et deux échelles RBF, sur CPU uniquement. Les bornes des longueurs sont réglables.

Le choix du noyau se fait indépendamment pour chaque sortie, par RMSE moyenne en régression et log-loss moyenne en classification. Le modèle final reste ajusté sur les 80 % d’apprentissage. Le test affiche RMSE/MAE/couverture à 95 % ou exactitude/F1 macro/log-loss/matrice de confusion. Les intervalles et probabilités ne garantissent pas une couverture ou une calibration parfaite.

Le sélecteur **Sortie à explorer** change les métriques et les graphiques. Les coupes fixent les autres entrées à leur médiane numérique ou leur modalité la plus fréquente. Les axes du graphique des données permettent de choisir n’importe quelle entrée. La matrice représente les variables numériques et catégorielles (histogrammes, effectifs et nuages de points). Le tableau affiche toutes les lignes en défilement virtualisé.

Dans la vue **Prédire**, le formulaire **Prédire un nouveau point** calcule toutes les sorties simultanément, sans fit. Les catégories utilisent des listes de modalités connues. Les valeurs numériques hors plage d’apprentissage sont signalées. Les prédictions de test sont exportables en CSV.

La navigation distingue **Configurer le modèle**, **Explorer les résultats** et **Prédire** et **Plan d’expériences**. **Nouveau modèle** permet de repartir sur un autre fichier. Les études restent en mémoire (10 imports et 10 analyses) jusqu’à l’arrêt du serveur. Un seul entraînement est exécuté à la fois.

## Plan d’expériences

Cette vue fonctionne **avant entraînement**, pour un plan LHS initial, et **après entraînement**, pour rechercher des configurations complémentaires.

- Pour chaque entrée numérique : bornes, valeurs entières éventuelles, loi uniforme, normale tronquée aux bornes (moyenne/écart-type) ou log-uniforme (bornes positives).
- Pour une catégorie : modalités autorisées et poids relatifs, normalisés automatiquement.
- Contraintes d’intervalle inclus/exclu, de catégories autorisées/interdites, de somme pondérée avec budget (égalité avec tolérance), et d’écart absolu entre deux valeurs. Comparaisons entre deux entrées ou avec une constante : `<`, `<=`, `>`, `>=`, `==`, `!=`. Toutes sont appliquées ensemble. Pour les catégories : égal/différent. Aucun texte Python n’est exécuté.
- Taille du plan et graine réglables. Les contraintes impossibles ou trop restrictives sont signalées. Une égalité exacte entre variables continues a une probabilité nulle par rejet ; utiliser une variable fixe ou un domaine adapté.

Le graphique de préparation affiche les points d’apprentissage, le test réservé et, après génération, les nouveaux points sur les mêmes axes.

Le LHS tire des quantiles indépendants par marge puis applique les quantiles inverses des lois choisies. Le plan initial renvoie jusqu’à 3 000 points distincts. La recherche par rejet est bornée à 50 000 tirages ou 64 lots. Elle peut renvoyer moins de points quand le domaine ne permet pas le nombre demandé.

L’enrichissement utilise un réservoir de 4 096 candidats par défaut (réglable jusqu’à 32 768), exclut toutes les configurations connues et propose 1 à 200 points. La priorité combine distance au plus proche voisin et incertitude, puis diversifie le lot. Les distances utilisent les entrées encodées et standardisées sur les observations connues. L’incertitude est l’écart-type latent (bruit d’observation soustrait) pour une régression, l’entropie pour une classification. Pour plusieurs sorties, on moyenne les rangs d’incertitude à poids égaux. Un filtre peut viser une plage ou une classe **prédite**, sans garantie sur la future mesure.

**Contraintes, exclusion, discrétisation et sélection peuvent modifier les strates LHS et les proportions marginales finales.** Les lois décrivent le tirage initial, pas une distribution garantie du lot retenu. Les contraintes ne définissent pas une loi jointe corrélée. Le bilan précise les tirages et rejets. Le CSV contient les entrées et toutes les sorties laissées vides pour les nouvelles mesures. Les estimations du modèle ne sont jamais ajoutées aux observations d’apprentissage.

## Sauvegarder, rouvrir, prédire dans Excel

**Sauvegarder le modèle** télécharge un `.gpmodel` avec les données, les réglages, les encodages et tous les postérieurs de régression/classification. **Ouvrir un modèle** les reconstitue sur CPU sans nouvel entraînement. Format ZIP versionné avec JSON et tableaux NumPy numériques, sans pickle/joblib. Les anciennes sauvegardes numériques v1 restent prises en charge ; les études mixtes utilisent v2. Maximum 100 Mo téléchargés à l’ouverture, 160 Mo décompressés. Le dernier plan généré est conservé dans la session serveur pour le rapport, pas dans la sauvegarde du modèle ; conserver son CSV séparément.

**Prédire depuis un fichier Excel** permet de choisir une feuille, les titres, les colonnes d’entrée et une colonne de destination pour chaque sortie. On peut créer plusieurs colonnes à la suite du tableau. Les résultats comprennent nombres et étiquettes de classe. Les cellules remplies sont conservées par défaut, individuellement par sortie. Les entrées incomplètes, nombres non finis ou modalités inconnues font ignorer la ligne, avec bilan des cellules écrites et numéros concernés. Limites : 20 Mo, 10 000 lignes par traitement et limites de colonnes Excel.

L’original reste intact. Les cellules choisies sont modifiées dans l’OOXML d’une copie ; les autres feuilles, styles et formules sont conservés. Une formule d’entrée doit avoir une valeur calculée enregistrée par Excel. Les sorties fusionnées sont refusées. Une nouvelle colonne n’étend pas automatiquement une table structurée Excel.

## Rapport PDF complet

Le bouton unique **Exporter le rapport complet (PDF)** exporte l’étude indépendamment de l’onglet affiché :

- sommaire cliquable avec numéros de pages et signets ;
- configuration, distributions et toutes les paires distinctes d’entrées ;
- toutes les sorties, métriques, prédictions de test et probabilités, coupes 1D, classements et paramètres ;
- dernier plan généré, lois, contraintes, configurations et estimations éventuelles ;
- diagnostics et annexe de toutes les données importées, réparties en groupes de colonnes si nécessaire.

Le plan seul peut aussi être exporté avant entraînement. Le rapport A4 paysage est construit sur le serveur avec graphiques vectoriels et tableaux, sans capture des vues ouvertes ni Chrome. Il peut devenir volumineux avec beaucoup de colonnes ou de lignes. Les PNG Plotly et CSV restent disponibles. L’ancien endpoint d’assemblage d’images reste accessible pour compatibilité, mais les boutons PDF individuels ont été retirés du site.

## GPU et accès SSH

Le CPU fonctionne sans PyTorch. CUDA accélère uniquement les régressions de la banque simple ; la classification et la banque élargie utilisent le CPU. Pour une étude mixte en mode CUDA, les sorties numériques utilisent le GPU et les sorties catégorielles le CPU, explicitement indiqués par sortie.

Dans **Réglages de l’analyse**, le diagnostic affiche la machine, le Python, la version PyTorch, sa version CUDA et le motif précis d’indisponibilité. Il vérifie un vrai calcul GP float64 : Cholesky, gradient et prédiction. On peut l’actualiser sans redémarrer. Aucun repli GPU vers CPU n’est caché. Les distances GPU sont calculées avec des matrices 2D pour éviter le tenseur mémoire en `n × n × nombre_d’entrées`.

Sur la machine SSH, installer une distribution PyTorch adaptée au pilote dans **le Python qui lance le serveur**, selon [l’installateur officiel PyTorch](https://pytorch.org/get-started/locally/). PyTorch n’est pas installé automatiquement dans les dépendances CPU. Tester depuis la racine :

```bash
website/.venv/bin/python website/tools/diagnose_gpu.py
```

Vérifier aussi `nvidia-smi`, l’allocation GPU (si ordonnanceur ou conteneur) et `CUDA_VISIBLE_DEVICES`. Pour accéder depuis un PC sans GPU, ouvrir un tunnel, par exemple :

```bash
ssh -L 8000:127.0.0.1:8000 utilisateur@serveur
```

Puis lancer le serveur sur la machine distante et ouvrir `http://127.0.0.1:8000` sur le PC client. Le GPU du PC client n’est pas utilisé. Le calcul sur une vraie carte CUDA ne peut être validé depuis une machine dépourvue de GPU ; les tests matériels sont conditionnels.

## Page d’explication et fichiers Python

**Comment ça fonctionne ?** ouvre `/static/fonctionnement.html` dans un nouvel onglet. Le schéma illustre types de sortie, nombre de modèles, plis et banque de noyaux. La page détaille encodage, régression/classification, sauvegarde, GPU/SSH, lois et contraintes LHS, acquisition et rapport PDF complet. La galerie présente huit exemples de régression 1D, pas des exemples de classification.

| Module | Fonction |
| --- | --- |
| `study.py` | Études à plusieurs sorties, types, partitions et validation croisée. |
| `typed_models.py` | Encodage, classification GP et regroupement des modèles indépendants. |
| `models.py`, `gpu_model.py` | Noyaux et régression GP CPU/CUDA. |
| `hardware.py` | Diagnostic matériel du serveur. |
| `design.py`, `acquisition.py` | Tirages, contraintes et sélection des prochaines simulations. |
| `model_io.py`, `bundle_io.py` | Archives portables v1/v2 sans code exécutable. |
| `report_export.py`, `excel_prediction.py` | Rapport complet et écriture des sorties dans Excel. |
| `api.py`, `app.py` | Import, calcul en arrière-plan, endpoints et serveur local. |

## Modifier le calcul Python

Le site n’exécute pas les fichiers `.ipynb`. Il importe directement les modules Python de `website` ; modifier un notebook ne change donc pas le site automatiquement.

| Fichier | Rôle / modifications |
| --- | --- |
| `models.py` → `noyaux_candidats(n_dim)` | Liste des noyaux, valeurs initiales et bornes de leurs hyperparamètres. Supprimer une entrée du dictionnaire `signaux` retire un candidat ; en ajouter une ajoute un candidat. Garder au moins un candidat. |
| `models.py` → `GPRWrapper` | Normalisation des entrées / sorties, choix CPU / CUDA, ajustement et prédiction dans les unités originales. |
| `study.py` → `prepare_study` / `run_study` | Partage apprentissage-test, nombre de plis, relances d’optimisation, métriques et données des graphiques. |
| `gpu_model.py` | Calcul matriciel PyTorch du notebook GPU. À adapter si un nouveau type de noyau doit fonctionner sur CUDA. |
| `api.py` | Import des fichiers, lancement des analyses, conservation des modèles et prédiction des nouveaux points. |

Exemple : dans `noyaux_candidats`, remplacer la ligne du RBF par ceci pour rechercher chaque longueur de corrélation entre 0,1 et 100 :

```python
"RBF (ARD)": RBF(
    length_scale=np.ones(n_dim),
    length_scale_bounds=(1e-1, 1e2),
),
```

Pour les Matérn, modifier leur deuxième argument `(1e-2, 1e3)`, ou utiliser également le mot-clé `length_scale_bounds`. Une paire de bornes s’applique à toutes les colonnes encodées ; une matrice `(n_dim, 2)` permet des bornes différentes pour chaque entrée, dans l’ordre des variables sélectionnées sur le site. Choisir des bornes strictement positives et une valeur initiale à l’intérieur.

Les entrées sont standardisées avant l’ajustement : ces ℓᵢ sont exprimés dans l’espace normalisé, pas dans les unités originales des Xᵢ. Dans ce code, RBF et Matérn sont anisotropes (une longueur par entrée), tandis que RationalQuadratic utilise une seule longueur commune. `ConstantKernel` règle l’amplitude et `WhiteKernel` le bruit ; leurs valeurs initiales et bornes sont à la fin de `noyaux_candidats`.

Après une modification Python : arrêter le serveur avec `Ctrl+C`, puis relancer `./website/start.sh` depuis la racine du projet. Actualiser la page, réimporter les données et réentraîner le modèle. Le serveur n’active pas le rechargement automatique ; ses anciens modèles en mémoire ne changent pas quand un fichier est modifié. Les modifications HTML/CSS/JS ne nécessitent qu’une actualisation du navigateur.

Le CPU utilise les noyaux scikit-learn. Le backend CUDA actuel accepte uniquement `ConstantKernel * (RBF / Matérn ν=1,5 ou 2,5 / RationalQuadratic) + WhiteKernel` : l’ajout d’un autre noyau demande aussi une implémentation dans `gpu_model.py`, ou l’utilisation du CPU.


## Vérifications

```bash
cd website
.venv/bin/python -m unittest discover -s tests -v
```

Les notebooks et `main.py` ne sont pas modifiés. Les modules du site sont autonomes. Une modification Python nécessite un redémarrage ; une modification HTML/CSS/JS nécessite une actualisation du navigateur.

Les formules du noyau retenu sont rendues en LaTeX avec [KaTeX](https://katex.org/docs/api), livré dans `static/vendor/katex` (licence MIT) pour fonctionner hors ligne. Les couleurs et le survol du tableau relient les symboles aux paramètres ; en classification multiclasse, un sélecteur affiche les paramètres de chaque classifieur un contre tous.
