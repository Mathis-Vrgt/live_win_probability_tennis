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

    # Finalisation
    return df.reset_index(drop=True)


if __name__ == "__main__":
    test_file_path = "/Users/mathisverguet/live_win_probability_tennis/data/raw/processed/charting-m-points-2010s.csv"
    try:
        df = load_data(test_file_path)
        print("Data loaded successfully:")
        df = df_complete_features(df)

        print(df.head(10))
        # nombre de lignes
        print(f"Nombre de lignes : {len(df)}")
        print(get_surface("Doha"))
        #afficher les colonnes
        print(df.columns)
    except Exception as e:
        print(f"Erreur : {e}")
