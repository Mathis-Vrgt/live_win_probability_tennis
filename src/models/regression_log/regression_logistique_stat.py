import pandas as pd
from sklearn.preprocessing import StandardScaler
from utils.features import load_data, df_complete_features
import statsmodels.api as sm


df = load_data("/Users/mathisverguet/live_win_probability_tennis/data/raw/processed/charting-m-points-2020s.csv")
df = df_complete_features(df)

# enlever les colonnes 2nd, 1st et Notes
df = df.drop(columns=['2nd', '1st', 'Notes'])
df = df.dropna().reset_index(drop=True)

# Split temporel propre non aléatoire (important pour les séries temporelles)
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


# Après analyse statistique, on ne garde que les features significatives
# on élève bien une catégorie par variable catégorielle

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


# modèle
X_train = X_train.astype(float)
y_train = y_train.astype(float)
print(X_train.shape, y_train.shape)
model = sm.Logit(y_train, X_train)
result = model.fit()
print(result.summary())
