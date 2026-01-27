import pandas as pd
import os
from config.config_load import TOURNAMENT_INFO, POINTS_BY_ROUND
from src.data_process.load_data import load_data
import numpy as np


def ajouter_noms_joueurs(df):
    """
    Extrait les noms des joueurs (avec les underscores) depuis le match_id.
    Ex: 'Roger_Federer' reste 'Roger_Federer'.
    """
    # On split l'ID pour récupérer les parties 4 et 5
    match_parts = df['match_id'].str.split('-', expand=True)

    df['Player1_Name'] = match_parts[4]
    df['Player2_Name'] = match_parts[5]

    return df


def select_tournament_data(df, tournament_name):
    """
    Select data for a specific tournament from the DataFrame.

    Parameters:
    df (pd.DataFrame): The DataFrame containing the data.
    tournament_name (str): The name of the tournament to filter by.

    Returns:
    pd.DataFrame: A DataFrame containing only the data for the specified tournament.
    """
    filtered_df = df[df['tournament_name'] == tournament_name]
    return filtered_df


# trouver le vainqueur du match
def get_match_winner(match_id, df):
    # Extraction du dernier point du match spécifique
    match_data = df[df['match_id'] == match_id]
    last_row = match_data.iloc[-1]

    return last_row['Player1_ID'] if last_row['PtWinner'] == 1 else last_row['Player2_ID']


def ajouter_colonne_vainqueur(df):
    # On s'assure que le DF est bien trié chronologiquement par match et par point
    # La colonne 2 (Pt) représente l'ordre des points
    df = df.sort_values(by=['match_id', 'Pt'])

    # On récupère la dernière ligne de chaque match (le point de la victoire)
    last_points = df.groupby('match_id').tail(1).copy()

    # Détermination du vainqueur
    winner_col = 'PtWinner' if 'PtWinner' in df.columns else df.columns[-1]

    last_points['Winner_ID'] = np.where(
        last_points[winner_col] == 1,
        last_points['Player1_ID'],
        last_points['Player2_ID']
    )

    # Création d'un dictionnaire pour mapper rapidement
    mapping = last_points.set_index('match_id')['Winner_ID'].to_dict()

    # Application à tout le DataFrame
    df['Winner'] = df['match_id'].map(mapping)

    return df


def breakpoint(row):
    """
    Détermine si le point en cours est une opportunité de break pour le joueur qui reçoit.
    """
    score = row['Pts']
    server = row['Svr']
    points = score_simple(score)
    server_points = points[0] if server == 1 else points[1]
    receiver_points = points[1] if server == 1 else points[0]

    # Opportunité de break si le receveur a 3 points (40) et le serveur moins de 3 points
    if receiver_points == 3 and server_points < 3 or (receiver_points == 4 and server_points == 3):
        return True
    return False

def is_set_point(row):
    """
    Détermine si le point actuel est une balle de set.
    """
    # 1. Utiliser row et non df (très important !)
    s1 = row['Simple_Score_Player1']
    s2 = row['Simple_Score_Player2']
    g1 = row['Gm1']
    g2 = row['Gm2']

    # 2. Qui est à un point de gagner le jeu ?
    # Rappel : 3 = 40, 4 = Avantage (AD)
    # Dans un tie-break, score_simple renvoie les points réels (6, 7, 8...)

    p1_set_pt = False
    p2_set_pt = False

    # Cas 1 : Jeu normal (P1 peut gagner le set)
    # P1 gagne le set s'il gagne le jeu et qu'il mène (5-4, 5-3, 5-2, 5-1, 5-0) OU s'il mène 6-5
    if (s1 >= 3 and s1 > s2):
        if (g1 == 5 and g2 <= 4) or (g1 == 6 and g2 == 5):
            p1_set_pt = True

    # Cas 2 : Jeu normal (P2 peut gagner le set)
    if (s2 >= 3 and s2 > s1):
        if (g2 == 5 and g1 <= 4) or (g2 == 6 and g1 == 5):
            p2_set_pt = True

    # Cas 3 : Tie-break (souvent Gm1=6 et Gm2=6)
    # Au tie-break, on gagne le set à 7 points (avec 2 points d'écart)
    if g1 == 6 and g2 == 6:
        if s1 >= 6 and s1 > s2: p1_set_pt = True
        if s2 >= 6 and s2 > s1: p2_set_pt = True

    return p1_set_pt or p2_set_pt

def get_surface(tournament_code):
    """Renvoie la surface du tournoi à partir de son code."""
    # On récupère le dictionnaire du tournoi, sinon un dict vide
    info = TOURNAMENT_INFO.get(tournament_code, {})
    # On renvoie la surface, ou 'Unknown' si elle n'est pas définie
    return info.get('surface', 'Unknown')


def get_level(tournament_code):
    """Renvoie le niveau du tournoi (ATP, GS, etc.) à partir de son code."""
    info = TOURNAMENT_INFO.get(tournament_code, {})
    return info.get('level', 'Unknown')


# Fonction pour attribuer les points
def get_points(df, tournament_type, round_reached, player):
    """Retourne les points gagnés ATP en fonction du tournoi et du round"""
    if round_reached == 'F' and df['Winner'] == player:
        return POINTS_BY_ROUND[tournament_type]['Winner']
    return POINTS_BY_ROUND[tournament_type][round_reached]


def extract_tournament_name(match_id):
    """Extrait 'Paris_Masters' de l'ID du match."""
    parts = match_id.split('-')
    if len(parts) >= 3:
        return parts[2]
    return "Unknown"


def score_simple(score):
    if pd.isna(score) or '-' not in str(score):
        return (0, 0)

    parts = str(score).split("-")
    # Petit dico pour le tennis standard
    tennis_map = {"0": 0, "15": 1, "30": 2, "40": 3, "AD": 4}

    def convert(p):
        p = p.strip()
        if p in tennis_map: return tennis_map[p]
        try:
            return int(p)  # Gère tous les chiffres de Tie-break (1, 2, ..., 20, etc.)
        except:
            return 0

    return (convert(parts[0]), convert(parts[1]))


def enrich_match_context(df):
    """
    On utilise les colonnes déjà créées dans df_complete_features.
    """
    # Conversion des scores (Vectorisé)
    scores_transformed = df['Pts'].apply(score_simple)
    df['Simple_Score_Player1'] = scores_transformed.apply(lambda x: x[0])
    df['Simple_Score_Player2'] = scores_transformed.apply(lambda x: x[1])

    # Identification des Breakpoints/set_points
    df['Is_Breakpoint'] = df.apply(breakpoint, axis=1)

    # Mapping Surface et Level (déjà extrait dans tournament_name)
    df['Surface'] = df['tournament_name'].map(get_surface)
    df['Tournament_Level'] = df['tournament_name'].map(get_level)

    return df

def calculate_elo(df, k_factor=32):
    """
    Calcule l'Elo en triant strictement par la date contenue dans le match_id.
    """
    # On trie par les 8 premiers caractères (YYYYMMDD) pour la chronologie
    # On ajoute Pt pour garder l'ordre interne si besoin, mais match_id suffit ici
    df_sorted = df.sort_values(by=['match_id'])

    # On récupère les matchs uniques dans l'ordre chronologique
    matches = df_sorted.drop_duplicates('match_id')

    elo_dict = {}
    elo_history = []

    for _, row in matches.iterrows():
        p1 = row['Player1_Name']
        p2 = row['Player2_Name']
        winner = row['Winner']

        r1 = elo_dict.get(p1, 1500)
        r2 = elo_dict.get(p2, 1500)

        # On stocke l'Elo AVANT le match
        elo_history.append({
            'match_id': row['match_id'],
            'Elo_P1': r1,
            'Elo_P2': r2,
            'Elo_Diff': r1 - r2
        })

        # Formule Elo
        expected_p1 = 1 / (1 + 10 ** ((r2 - r1) / 400))
        actual_p1 = 1 if winner == p1 else 0

        # Mise à jour pour les prochains matchs
        elo_dict[p1] = r1 + k_factor * (actual_p1 - expected_p1)
        elo_dict[p2] = r2 + k_factor * ((1 - actual_p1) - (1 - expected_p1))

    return pd.DataFrame(elo_history)


def calculate_live_momentum(df):
    """
    Calcule les statistiques de performance en direct durant le match de manière optimisée.
    """

    df['p1_served_and_won'] = ((df['Svr'] == 1) & (df['PtWinner'] == 1)).astype(int)
    df['p1_is_serving'] = (df['Svr'] == 1).astype(int)

    df['p2_served_and_won'] = ((df['Svr'] == 2) & (df['PtWinner'] == 2)).astype(int)
    df['p2_is_serving'] = (df['Svr'] == 2).astype(int)

    # 2. Somme cumulée par match (vectorisé)
    # On groupe par match_id pour ne pas mélanger les stats d'un match à l'autre
    group = df.groupby('match_id')

    df['p1_pts_won_on_serve'] = group['p1_served_and_won'].cumsum()
    df['p1_total_serve_pts'] = group['p1_is_serving'].cumsum()

    df['p2_pts_won_on_serve'] = group['p2_served_and_won'].cumsum()
    df['p2_total_serve_pts'] = group['p2_is_serving'].cumsum()

    # 3. Calcul des ratios (Momentum de service)
    # On utilise .replace(0, 1) pour éviter la division par zéro au premier point
    df['p1_serve_win_rate'] = df['p1_pts_won_on_serve'] / df['p1_total_serve_pts'].replace(0, 1)
    df['p2_serve_win_rate'] = df['p2_pts_won_on_serve'] / df['p2_total_serve_pts'].replace(0, 1)

    # Nettoyage des colonnes temporaires
    df = df.drop(columns=['p1_served_and_won', 'p1_is_serving', 'p2_served_and_won', 'p2_is_serving', 'p1_pts_won_on_serve', 'p1_total_serve_pts', 'p2_pts_won_on_serve', 'p2_total_serve_pts'])

    return df


def add_dominance_features(df):
    # Appliquer to_numeric avec errors='coerce' transforme les erreurs en NaN
    # Ensuite on remplit les NaN par 0
    df['Set1'] = pd.to_numeric(df['Set1'], errors='coerce').fillna(0).astype(int)
    df['Set2'] = pd.to_numeric(df['Set2'], errors='coerce').fillna(0).astype(int)
    df['Gm1'] = pd.to_numeric(df['Gm1'], errors='coerce').fillna(0).astype(int)
    df['Gm2'] = pd.to_numeric(df['Gm2'], errors='coerce').fillna(0).astype(int)

    df['set_diff'] = df['Set1'] - df['Set2']
    df['game_diff'] = df['Gm1'] - df['Gm2']
    return df




def calculate_historical_momentum(df, window=5):
    """
    Calcule la forme récente de manière sécurisée contre les valeurs manquantes.
    """
    # 1. On nettoie : on ne garde que les lignes où on a un vainqueur identifié
    # Si Winner est NaN, on ne peut pas calculer de streak fiable
    match_results = df[['match_id', 'Player1_Name', 'Player2_Name', 'Winner']].drop_duplicates().copy()
    match_results = match_results.dropna(subset=['Winner'])

    match_results['date'] = match_results['match_id'].str[:8]
    match_results = match_results.sort_values('date')

    def get_streak(player_name, current_date):
        # Filtre les matchs passés
        past_matches = match_results[
            (match_results['date'] < current_date) &
            ((match_results['Player1_Name'] == player_name) | (match_results['Player2_Name'] == player_name))
        ].tail(window)

        if len(past_matches) == 0:
            return 0.5  # Valeur neutre (50/50) plutôt que 0 pour éviter les biais

        # Somme des victoires (booléen vers float, pas de conversion int ici)
        wins = (past_matches['Winner'] == player_name).sum()
        return float(wins) / len(past_matches)

    # 2. Application
    unique_matches = match_results[['match_id', 'Player1_Name', 'Player2_Name', 'date']].copy()

    # On s'assure que les noms sont bien des strings pour éviter les erreurs de comparaison
    unique_matches['Player1_Name'] = unique_matches['Player1_Name'].astype(str)
    unique_matches['Player2_Name'] = unique_matches['Player2_Name'].astype(str)

    unique_matches['p1_recent_form'] = unique_matches.apply(
        lambda x: get_streak(x['Player1_Name'], x['date']), axis=1)
    unique_matches['p2_recent_form'] = unique_matches.apply(
        lambda x: get_streak(x['Player2_Name'], x['date']), axis=1)

    # On remplit les éventuels NaN restants par sécurité
    unique_matches['p1_recent_form'] = unique_matches['p1_recent_form'].fillna(0.5)
    unique_matches['p2_recent_form'] = unique_matches['p2_recent_form'].fillna(0.5)

    return unique_matches[['match_id', 'p1_recent_form', 'p2_recent_form']]


def df_complete_features(df):
    # Nettoyage : On ne garde que les vrais points
    df = df[df['Pts'].str.contains('-', na=False)].copy()

    # Extraction unique (On le fait ici, donc plus besoin dans enrich_match_context)
    match_parts = df['match_id'].str.split('-', expand=True)
    df['Round_Code'] = match_parts[3]
    df['Player1_ID'] = match_parts[4]
    df['Player2_ID'] = match_parts[5]
    df['tournament_name'] = match_parts[2]

    # Calculs
    df = ajouter_noms_joueurs(df)
    df = ajouter_colonne_vainqueur(df)
    df = enrich_match_context(df)
    df['Is_Set_Point'] = df.apply(is_set_point, axis=1)

    # elo
    print("Calcul de l'Elo en cours...")
    elo_map = calculate_elo(df)
    df = df.merge(elo_map, on='match_id', how='left')

    # momentum live
    print("Calcul du momentum en cours...")
    df = calculate_live_momentum(df)
    df = add_dominance_features(df)
    historical_momentum_df = calculate_historical_momentum(df)
    # On fusionne ce résultat avec le df principal
    df = df.merge(historical_momentum_df, on='match_id', how='left')

    # 3.On remplit les cases vides par 0.5 (neutre) pour les nouveaux joueurs
    df['p1_recent_form'] = df['p1_recent_form'].fillna(0.5)
    df['p2_recent_form'] = df['p2_recent_form'].fillna(0.5)
    print(df['p1_recent_form'].quantile([0.25, 0.5, 0.75, 0.99]))

    # Finalisation
    return df.reset_index(drop=True)




if __name__ == "__main__":
    test_file_path = "/Users/mathisverguet/live_win_probability_tennis/data/raw/processed/charting-m-points-2010s.csv"
    try:
        df = load_data(test_file_path)
        print("Data loaded successfully:")
        df = df_complete_features(df)

        print(df.tail(10))
        # nombre de lignes
        print(f"Nombre de lignes : {len(df)}")

    except Exception as e:
        print(f"Erreur : {e}")
