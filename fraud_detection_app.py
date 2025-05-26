# fraud_detection_app.py
import streamlit as st
import pandas as pd
import numpy as np
import os
import joblib

from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from xgboost import XGBClassifier
from imblearn.over_sampling import SMOTE
from sklearn.metrics import classification_report, confusion_matrix, roc_auc_score, f1_score, roc_curve, auc, ConfusionMatrixDisplay
import matplotlib.pyplot as plt
import seaborn as sns

# Ajuster taille police globale pour les graphiques
plt.rcParams.update({'font.size': 12})

# ----- Utility functions -----
def save_model(model, model_name):
    joblib.dump((model, model_name), f"{model_name}.pkl")

def load_model(model_name):
    path = f"{model_name}.pkl"
    if os.path.exists(path):
        return joblib.load(path)[0]
    return None

# ----- Streamlit App -----
st.set_page_config(page_title="Fraud Detection App", layout="wide")
st.title("💳 Fraud Detection App")

st.sidebar.header("1. Upload Transaction Dataset")
uploaded_file = st.sidebar.file_uploader("Upload a CSV file", type=["csv"])

if uploaded_file is not None:
    df = pd.read_csv(uploaded_file)

    st.subheader("ℹ️ Informations sur le Dataset")
    st.write(f"Nombre de lignes : {df.shape[0]}")
    st.write(f"Nombre de colonnes : {df.shape[1]}")
    st.write("Types de colonnes :")
    st.write(df.dtypes)
    st.write("Résumé statistique des colonnes numériques :")
    st.write(df.describe())

    st.subheader("📊 Aperçu du Dataset")
    st.dataframe(df.head())

    if 'isFraud' not in df.columns:
        st.warning("The dataset must contain a column named 'isFraud' as the target variable.")
    else:
        st.subheader("📉 Distribution des classes (avant SMOTE)")
        fig1, ax1 = plt.subplots(figsize=(6, 4))
        sns.countplot(x='isFraud', data=df, ax=ax1)
        ax1.set_title("Nombre de transactions frauduleuses vs normales")
        st.pyplot(fig1, use_container_width=False)
        st.caption("Ce graphique montre combien de transactions sont frauduleuses (1) vs normales (0).")

        st.subheader("🔗 Matrice de corrélation")
        numeric_df = df.select_dtypes(include=[np.number])
        fig2, ax2 = plt.subplots(figsize=(8, 6))
        corr = numeric_df.corr()
        sns.heatmap(corr, cmap='coolwarm', annot=False, ax=ax2)
        st.pyplot(fig2, use_container_width=False)
        st.caption("Cette carte thermique montre la corrélation entre les variables numériques.")

        df = df.drop(['nameOrig', 'nameDest'], axis=1, errors='ignore')
        if df['type'].dtype == 'object':
            df = pd.get_dummies(df, columns=['type'])

        X = df.drop('isFraud', axis=1)
        y = df['isFraud']

        sm = SMOTE(random_state=42)
        X_resampled, y_resampled = sm.fit_resample(X, y)
        X_train, X_test, y_train, y_test = train_test_split(X_resampled, y_resampled, test_size=0.2, random_state=42)

        models = {
            "Random Forest": RandomForestClassifier(n_estimators=100, class_weight="balanced", random_state=42),
            "Logistic Regression": LogisticRegression(max_iter=1000, class_weight="balanced", random_state=42),
            "XGBoost": XGBClassifier(use_label_encoder=False, eval_metric='logloss', scale_pos_weight=1)
        }

        st.sidebar.header("2. Select Model")
        selected_model_name = st.sidebar.selectbox("Choose model", list(models.keys()))

        model_file_name = selected_model_name.lower().replace(" ", "_") + ".pkl"
        model = load_model(model_file_name)

        if model is not None:
            st.success(f"✅ Modèle pré-entraîné {selected_model_name} chargé avec succès.")
        elif st.sidebar.button("Entraîner le modèle"):
            with st.spinner("⏳ Entraînement en cours..."):
                model = models[selected_model_name]
                model.fit(X_train, y_train)
                save_model(model, selected_model_name.lower().replace(" ", "_"))

                y_pred = model.predict(X_test)

                st.subheader(f"📈 Rapport d’évaluation pour {selected_model_name}")
                st.text(classification_report(y_test, y_pred))
                st.write("ROC AUC Score:", roc_auc_score(y_test, y_pred))
                st.write("F1 Score:", f1_score(y_test, y_pred))

                cm = confusion_matrix(y_test, y_pred)
                fig_cm, ax_cm = plt.subplots(figsize=(6, 4))
                sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', ax=ax_cm)
                ax_cm.set_title("Matrice de confusion")
                st.pyplot(fig_cm, use_container_width=False)
                st.caption("🔢 Montre le nombre de vraies/faux positifs et négatifs. Permet de voir où le modèle se trompe.")

                if hasattr(model, 'feature_importances_'):
                    st.subheader("📌 Importance des variables")
                    importances = pd.Series(model.feature_importances_, index=X.columns)
                    importances = importances.sort_values(ascending=False)[:20]
                    fig3, ax3 = plt.subplots(figsize=(6, 4))
                    sns.barplot(x=importances.values, y=importances.index, ax=ax3)
                    ax3.set_title("Top 20 variables les plus importantes")
                    st.pyplot(fig3, use_container_width=False)
                    st.caption("Ce graphique montre quelles variables ont le plus contribué aux décisions du modèle.")

                if hasattr(model, "predict_proba"):
                    y_score = model.predict_proba(X_test)[:, 1]
                    fpr, tpr, _ = roc_curve(y_test, y_score)
                    roc_auc = auc(fpr, tpr)
                    st.subheader("📉 Courbe ROC")
                    fig4, ax4 = plt.subplots(figsize=(6, 4))
                    ax4.plot(fpr, tpr, label=f"ROC curve (AUC = {roc_auc:.2f})")
                    ax4.plot([0, 1], [0, 1], 'k--')
                    ax4.set_xlabel("False Positive Rate")
                    ax4.set_ylabel("True Positive Rate")
                    ax4.set_title("Receiver Operating Characteristic")
                    ax4.legend(loc="lower right")
                    st.pyplot(fig4, use_container_width=False)
                    st.caption("La courbe ROC montre la capacité du modèle à distinguer les classes. Plus l'AUC est proche de 1, meilleur est le modèle.")

                cm_norm = confusion_matrix(y_test, y_pred, normalize='true')
                disp = ConfusionMatrixDisplay(confusion_matrix=cm_norm, display_labels=model.classes_)
                st.subheader("Matrice de confusion (normalisée)")
                fig5, ax5 = plt.subplots(figsize=(6, 4))
                disp.plot(cmap='Blues', ax=ax5)
                st.pyplot(fig5, use_container_width=False)
                st.caption("Cette version normalisée montre les taux de bonnes prédictions pour chaque classe.")

                if 'amount' in df.columns:
                    st.subheader("💰 Distribution des montants par classe de fraude")
                    fig6, ax6 = plt.subplots(figsize=(6, 4))
                    sns.kdeplot(data=df[df['isFraud'] == 0], x="amount", label="Non frauduleux", fill=True)
                    sns.kdeplot(data=df[df['isFraud'] == 1], x="amount", label="Frauduleux", fill=True, color="red")
                    ax6.set_xlim([0, df['amount'].quantile(0.99)])
                    ax6.legend()
                    st.pyplot(fig6, use_container_width=False)
                    st.caption("Comparaison des montants pour les transactions frauduleuses et non frauduleuses.")

                st.success("✅ Modèle entraîné, évalué et sauvegardé !")

        st.sidebar.header("3. Predict on New Data")
        predict_file = st.sidebar.file_uploader("Upload new data for prediction (same format)", type=["csv"], key="predict")

        if predict_file is not None and model and st.sidebar.button("Predict"):
            predict_df = pd.read_csv(predict_file)
            original = predict_df.copy()

            try:
                predict_df = predict_df.drop(['nameOrig', 'nameDest'], axis=1, errors='ignore')
                if 'type' in predict_df.columns and predict_df['type'].dtype == 'object':
                    predict_df = pd.get_dummies(predict_df, columns=['type'])
                for col in X.columns:
                    if col not in predict_df.columns:
                        predict_df[col] = 0
                predict_df = predict_df[X.columns]

                preds = model.predict(predict_df)
                original['isFraud_pred'] = preds
                st.subheader("🔍 Transactions Prédites comme Frauduleuses")
                st.dataframe(original[original['isFraud_pred'] == 1])

                csv = original.to_csv(index=False).encode('utf-8')
                st.download_button("🗕️ Télécharger les Prédictions", data=csv, file_name="fraud_predictions.csv", mime="text/csv")
            except Exception as e:
                st.error(f"Erreur lors de la prédiction : {e}")
                st.write("Colonnes attendues :", X.columns.tolist())
                st.write("Colonnes reçues :", predict_df.columns.tolist())
else:
    st.info("📂 En attente de l'importation d'un fichier CSV...")
