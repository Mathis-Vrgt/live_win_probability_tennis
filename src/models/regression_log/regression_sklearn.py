# python -m src.models.regression_log.regression_sklearn
import pandas as pd
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from utils.features import load_data, df_complete_features
import matplotlib.pyplot as plt


df = load_data("/Users/mathisverguet/live_win_probability_tennis/data/raw/processed/charting-m-points-2020s.csv")
df = df_complete_features(df)

# enlever les colonnes 2nd, 1st et Notes
df = df.drop(columns=['2nd', '1st', 'Notes'])
df = df.dropna().reset_index(drop=True)

# Split temporel propre
df_train = df[df['year'] <= 2023].copy()
df_test = df[df['year'] > 2024].copy()

# Liste des colonnes à supprimer
cols_to_drop = [
    'match_id', 'Player1_ID', 'Player2_ID', 'Player1_Name', 'Player2_Name',
    'tournament_name', 'Winner', 'PtWinner', 'year', 'Round_Code',
    'Pts'
]

# one hot Encodage rapide de la Surface, du service, et du tournoi
X_train = pd.get_dummies(df_train.drop(columns=cols_to_drop + ['Winner_Binary']), columns=['Surface', 'Tournament_Level', 'Svr'])
X_test = pd.get_dummies(df_test.drop(columns=cols_to_drop + ['Winner_Binary']), columns=['Surface', 'Tournament_Level', 'Svr'])


# Après analyse statistique, on ne garde que les features significatives (cf regression_logistique_stat.py)
significant_features = ['Elo_Diff', 'Pt', 'set_diff', 'game_diff', 'p1_serve_win_rate', 'p2_serve_win_rate',
                        'TbSet', 'Gm#', 'p1_recent_form', 'p2_recent_form', 'Svr_1', 'Simple_Score_Player1',
                        'Simple_Score_Player2', 'p1_total_matches', 'p2_total_matches',
                        'Surface_Clay', 'Surface_Grass',
                        'Tournament_Level_M1000', 'Tournament_Level_ATP500', 'Tournament_Level_ATP250',
                        'Tournament_Level_GS']


X_train = X_train[significant_features]
X_test = X_test[significant_features]


# Alignement des colonnes
X_train, X_test = X_train.align(X_test, join='left', axis=1, fill_value=0)

y_train = df_train['Winner_Binary']
y_test = df_test['Winner_Binary']

# Normalisation
features_a_normaliser = [
    'Elo_Diff',
    'p1_serve_win_rate', 'p2_serve_win_rate',
    'p1_recent_form', 'p2_recent_form',
    'p1_total_matches', 'p2_total_matches',
    'game_diff', 'set_diff', 'Pt', 'Gm#'
]

scaler = StandardScaler()

X_train.loc[:, features_a_normaliser] = scaler.fit_transform(X_train[features_a_normaliser])
X_test.loc[:, features_a_normaliser] = scaler.transform(X_test[features_a_normaliser])


print(X_train.shape, y_train.shape)

# 4. On relance le modèle de sklearn qui est plus performant
model = LogisticRegression(max_iter=1000)
model.fit(X_train, y_train)
model.score(X_train, y_train)

print(f"Score Train: {model.score(X_train, y_train):.3f}")
print(f"Score Test: {model.score(X_test, y_test):.3f}")

############################### Affichage de la live win probability ####################################

# Choisir un match spécifique dans le jeu de test
target_match = "20250608-M-Roland_Garros-F-Jannik_Sinner-Carlos_Alcaraz"

# Extraire les points de ce match et leurs caractéristiques
match_indices = df_test[df_test['match_id'] == target_match].index
X_match = X_test.loc[match_indices]

# Prédire les probabilités (colonne 1 = Probabilité que Player 1 gagne)
probabilities = model.predict_proba(X_match)[:, 1]

# Récupérer les infos pour l'affichage (Score, Joueurs)
match_info = df_test[df_test['match_id'] == target_match].iloc[0]
p1_name = match_info['Player1_Name']
p2_name = match_info['Player2_Name']


plt.figure(figsize=(12, 6))
plt.plot(range(len(probabilities)), probabilities, label='Win Probability P1', color='blue', linewidth=2)

# Ligne d'équilibre (50%)
plt.axhline(y=0.5, color='red', linestyle='--', alpha=0.5)


plt.ylim(0, 1)
plt.title(f"Live Win Probability: {p1_name} vs {p2_name}")
plt.xlabel("Points joués")
plt.ylabel(f"Probabilité de victoire ({p1_name})")
plt.grid(True, alpha=0.3)
plt.legend()


plt.savefig('win_probability_curve.png')
print(f"Courbe de probabilité générée pour le match : {target_match}")
