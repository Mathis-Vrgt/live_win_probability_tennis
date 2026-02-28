# python -m src.models.xgboost_model.predict_xgboost_clean
import pandas as pd
import numpy as np
import xgboost as xgb
from sklearn.metrics import log_loss, brier_score_loss, accuracy_score
from utils.features import load_data, df_complete_features
import matplotlib.pyplot as plt
from sklearn.calibration import calibration_curve
import seaborn as sns
from utils.utils_df import selectionner_match_fort_leverage

# 1. Chargement et Nettoyage
print("Chargement des données...")
df = load_data("/Users/mathisverguet/live_win_probability_tennis/data/raw/processed/charting-m-points-2020s.csv")
df = df_complete_features(df)
df_2 = load_data("/Users/mathisverguet/live_win_probability_tennis/data/raw/processed/charting-m-points-2010s.csv")
df_2 = df_complete_features(df_2)
df3 = load_data("/Users/mathisverguet/live_win_probability_tennis/data/raw/processed/charting-m-points-to-2009.csv")
df3 = df_complete_features(df3)
# ajout du df1, df2 et df3
df = pd.concat([df, df_2, df3], ignore_index=True)

# Nettoyage des colonnes inutiles
cols_to_drop_raw = ['2nd', '1st', 'Notes']
df = df.drop(columns=[c for c in cols_to_drop_raw if c in df.columns])
df = df.dropna(subset=['Winner_Binary']).reset_index(drop=True)

# 2. Split train/val/test
df_train = df[df['year'] <= 2015].copy()
df_val = df[(df['year'] > 2015) & (df['year'] <= 2019)].copy()
df_test = df[df['year'] >= 2020].copy()

print(f"Train: {len(df_train)} points | Val: {len(df_val)} points | Test: {len(df_test)} points")

significant_features = [
    'Elo_Diff', 'Pt', 'set_diff', 'game_diff',
    'Gm#', 'p1_recent_form', 'p2_recent_form', 'Svr_1', 'Simple_Score_Player1',
    'Simple_Score_Player2', 'p1_total_matches', 'p2_total_matches',
    'Surface_Clay', 'Surface_Grass', 'Surface_Hard',
    'Tournament_Level_GS', 'Tournament_Level_M1000', 'Tournament_Level_ATP500', 'Tournament_Level_ATP250',
    "Elo_Surface_Diff", "Svr_2"
]

# Colonnes de base pour le get_dummies
cols_to_drop = ['match_id', 'Player1_Name', 'Player2_Name', 'year', 'Winner_Binary', 'Round_Code', 'Pts', 'Winner', 'PtWinner', 'tournament_name', 'Player1_ID', 'Player2_ID']


def prepare_matrices(data, features_list):
    # Encodage One-Hot
    X = pd.get_dummies(data.drop(columns=[c for c in cols_to_drop if c in data.columns]),
                       columns=['Surface', 'Tournament_Level', 'Svr'])

    X = X.reindex(columns=features_list, fill_value=0)
    y = data['Winner_Binary']
    return X, y


X_train, y_train = prepare_matrices(df_train, significant_features)
X_val, y_val = prepare_matrices(df_val, significant_features)
X_test, y_test = prepare_matrices(df_test, significant_features)

# 3. Modèle XGBoost
model = xgb.XGBClassifier(
    n_estimators=2000,
    max_depth=3,
    learning_rate=0.01,
    subsample=1,
    colsample_bytree=0.6,
    eval_metric='logloss',
    early_stopping_rounds=100,
    random_state=42
)

print("Entraînement en cours...")
model.fit(
    X_train, y_train,
    eval_set=[(X_val, y_val)],
    verbose=False
)

y_pred_proba = model.predict_proba(X_test)[:, 1]
y_pred_bin = model.predict(X_test)
probas = model.predict_proba(X_test)

L = []

for match_id, match_df in df_test.groupby('match_id'):
    X_match = X_test.loc[match_df.index]

    probas = model.predict_proba(X_match)

    # 2. Identifier les indices de "leverage" (indices locaux : 0, 1, 2...)
    indices_relatifs = selectionner_match_fort_leverage(probas)

    # 3. Conversion en indices GLOBAUX de df_test
    # On utilise .index du groupe pour mapper l'index relatif à l'index réel
    indices_globaux = match_df.index[indices_relatifs].tolist()

    # 4. Ajouter à la liste globale (extend pour une liste plate)
    L.extend(indices_globaux)

# leverage
X_test_leverage = X_test.loc[L]
y_test_leverage = y_test.loc[L]

# Prédiction des probabilités pour ce sous-ensemble
probas_leverage = model.predict_proba(X_test_leverage)[:, 1]

# Calcul des scores
bs_leverage = brier_score_loss(y_test_leverage, probas_leverage)
ll_leverage = log_loss(y_test_leverage, probas_leverage)

# ---  Affichage des résultats ---
print(f"Résultats High Leverage (n={len(L)})")
print(f"Brier Score (Leverage): {bs_leverage:.3f}")
print(f"Log Loss (Leverage): {ll_leverage:.3f}")


print("\nRésultats sur l'ensemble de test")
print(f"Accuracy:    {accuracy_score(y_test, y_pred_bin):.4f}")
print(f"Log Loss:    {log_loss(y_test, y_pred_proba):.4f}")
print(f"Brier Score: {brier_score_loss(y_test, y_pred_proba):.4f}")

# 5. Visualisation
plt.figure(figsize=(10, 6))
xgb.plot_importance(model, max_num_features=15, importance_type='gain')
plt.title('Importance des Features (Gain)')
plt.savefig('xgboost_clean_importance.png')


def plot_calibration(y_true, y_probs, model_name):
    # prob_true : la fréquence réelle observée
    # prob_pred : la probabilité moyenne prédite par le modèle
    prob_true, prob_pred = calibration_curve(y_true, y_probs, n_bins=10)

    plt.plot(prob_pred, prob_true, marker='o', linewidth=1, label=model_name)


plt.figure(figsize=(8, 8))

# On trace la diagonale de référence
plt.plot([0, 1], [0, 1], linestyle='--', color='gray', label='Parfaite Calibration')

# On ajoute tes modèles (exemple avec tes résultats XGBoost)
plot_calibration(y_test, y_pred_proba, "Xgboost")


plt.xlabel('Probabilité Prédite')
plt.ylabel('Fréquence Réelle de Victoire')
plt.title('Courbe de Calibration : Xgboost vs Réalité')
plt.legend()
plt.grid(True, alpha=0.3)
plt.show()


def plot_discrimination(y_true, y_probs, model_name):
    plt.figure(figsize=(10, 6))
    # Distribution pour les perdants (0)
    plt.plot([0, 1], [0, 1], "k--", label="Parfaite Calibration", alpha=0.6)
    sns.kdeplot(y_probs[y_true == 0], label='Réalité : Défaite', shade=True, color='red')
    # Distribution pour les gagnants (1)
    sns.kdeplot(y_probs[y_true == 1], label='Réalité : Victoire', shade=True, color='blue')

    plt.title(f'Capacité de Discrimination - {model_name}')
    plt.xlabel('Probabilité de victoire prédite')
    plt.ylabel('Densité de matchs')
    plt.legend()
    plt.show()
