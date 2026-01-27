import pandas as pd
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from utils.features import load_data, df_complete_features
import statsmodels.api as sm


# 1. Chargement (Assure-toi que les chemins sont corrects)
df = load_data("/Users/mathisverguet/live_win_probability_tennis/data/raw/processed/charting-m-points-2020s.csv")
df = df_complete_features(df)
# enlever les colonnes 2nd, 1st et Notes
df = df.drop(columns=['2nd', '1st', 'Notes'])
df = df.dropna().reset_index(drop=True)

# 2. Split temporel propre
df_train = df[df['year'] <= 2023].copy()
df_test = df[df['year'] > 2024].copy()
# 3. Liste des colonnes à supprimer (Correction des noms)
# On retire tout ce qui est ID, Texte ou Target pour ne garder que le prédictif
cols_to_drop = [
    'match_id', 'Player1_ID', 'Player2_ID', 'Player1_Name', 'Player2_Name',
    'tournament_name', 'Winner', 'PtWinner', 'year', 'Round_Code',
    'Pts'
]

# 4. Encodage rapide de la Surface (Exemple)
# La logistique a besoin de chiffres. On transforme 'Clay', 'Hard', 'Grass' en colonnes 0/1
X_train = pd.get_dummies(df_train.drop(columns=cols_to_drop + ['Winner_Binary']), columns=['Surface', 'Tournament_Level'])
X_test = pd.get_dummies(df_test.drop(columns=cols_to_drop + ['Winner_Binary']), columns=['Surface', 'Tournament_Level'])

# Alignement des colonnes (pour être sûr d'avoir les mêmes colonnes dans les deux sets)
X_train, X_test = X_train.align(X_test, join='left', axis=1, fill_value=0)

y_train = df_train['Winner_Binary']
y_test = df_test['Winner_Binary']

# 5. Normalisation
features_a_normaliser = [
    'Elo_P1', 'Elo_P2', 'Elo_Diff',
    'p1_serve_win_rate', 'p2_serve_win_rate',
    'p1_recent_form', 'p2_recent_form',
    'p1_total_matches', 'p2_total_matches',
    'game_diff', 'set_diff', 'Pt', 'Gm#'
]

scaler = StandardScaler()

# On utilise .loc pour éviter le SettingWithCopyWarning
X_train.loc[:, features_a_normaliser] = scaler.fit_transform(X_train[features_a_normaliser])
X_test.loc[:, features_a_normaliser] = scaler.transform(X_test[features_a_normaliser])


# 6. Entraînement
model = LogisticRegression(max_iter=1000)
model.fit(X_train, y_train)

print(f"Score Train: {model.score(X_train, y_train):.3f}")
print(f"Score Test: {model.score(X_test, y_test):.3f}")



import matplotlib.pyplot as plt

# 1. Choisir un match spécifique dans le jeu de test
match_ids_test = df_test['match_id'].unique()
target_match = "20250608-M-Roland_Garros-F-Jannik_Sinner-Carlos_Alcaraz"

# 2. Extraire les points de ce match et leurs caractéristiques
match_indices = df_test[df_test['match_id'] == target_match].index
X_match = X_test.loc[match_indices]

# 3. Prédire les probabilités (colonne 1 = Probabilité que Player 1 gagne)
probabilities = model.predict_proba(X_match)[:, 1]

# 4. Récupérer les infos pour l'affichage (Score, Joueurs)
match_info = df_test[df_test['match_id'] == target_match].iloc[0]
p1_name = match_info['Player1_Name']
p2_name = match_info['Player2_Name']

# 5. Tracer la courbe
plt.figure(figsize=(12, 6))
plt.plot(range(len(probabilities)), probabilities, label='Win Probability P1', color='blue', linewidth=2)

# Ligne d'équilibre (50%)
plt.axhline(y=0.5, color='red', linestyle='--', alpha=0.5)

# Cosmétique du graphique
plt.ylim(0, 1)
plt.title(f"Live Win Probability: {p1_name} vs {p2_name}")
plt.xlabel("Points joués")
plt.ylabel(f"Probabilité de victoire ({p1_name})")
plt.grid(True, alpha=0.3)
plt.legend()

# Sauvegarde de l'image
plt.savefig('win_probability_curve.png')
print(f"Courbe de probabilité générée pour le match : {target_match}")
