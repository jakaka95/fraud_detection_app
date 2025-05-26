# 🛡️ Fraud Detection App - Streamlit

Application interactive de détection de fraudes basée sur des modèles de machine learning, développée avec Streamlit.

## 🎯 Objectif

Détecter automatiquement les transactions financières frauduleuses à l’aide de modèles supervisés (Random Forest, Régression Logistique, XGBoost) entraînés sur un jeu de données déséquilibré.

## 🔍 Fonctionnalités principales

- 📂 Chargement dynamique de jeux de données `.csv`
- 📊 Statistiques descriptives et visualisations (corrélation, distribution, etc.)
- ⚖️ Équilibrage des classes avec **SMOTE**
- 🤖 Entraînement et évaluation de plusieurs modèles de classification
- 💾 Sauvegarde et chargement de modèles entraînés
- 🧮 Prédictions sur de nouvelles transactions
- ⬇️ Téléchargement des résultats prédits

## 🧠 Modèles disponibles

- Random Forest
- Régression Logistique
- XGBoost

## 📈 Métriques affichées

- Rapport de classification
- Matrice de confusion (brute & normalisée)
- Courbe ROC / AUC
- Importance des variables

## 📁 Format de données attendu

Le fichier CSV utilisé doit inclure :
- Une colonne `isFraud` (0 = normal, 1 = fraude)
- Des colonnes numériques et une colonne catégorielle `type`
- Les colonnes `nameOrig` et `nameDest` sont supprimées automatiquement

Exemple de colonnes :
```csv
step,type,amount,oldbalanceOrg,newbalanceOrig,oldbalanceDest,newbalanceDest,isFraud
1,TRANSFER,181.0,181.0,0.0,21182.0,0.0,1
