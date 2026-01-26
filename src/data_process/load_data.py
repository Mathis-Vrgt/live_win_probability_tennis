import pandas as pd
import os
from config.config_load import TOURNAMENT_INFO, POINTS_BY_ROUND


def load_data(file_path):
    """
    Load data from a CSV file into a pandas DataFrame.

    Parameters:
    file_path (str): The path to the CSV file.

    Returns:
    pd.DataFrame: The loaded data as a pandas DataFrame.
    """
    if not os.path.exists(file_path):
        raise FileNotFoundError(f"The file at {file_path} does not exist.")

    try:
        data = pd.read_csv(file_path)
        return data
    except Exception as e:
        raise RuntimeError(f"An error occurred while loading the data: {e}")

# test the function


#if __name__ == "__main__":
#    test_file_path = "/Users/mathisverguet/live_win_probability_tennis/data/raw/processed/charting-m-points-2010s.csv"
#    try:
#        df = load_data(test_file_path)
#        print("Data loaded successfully:")
#        print(df.head())
#    except Exception as e:
#        print(e)

def ajout_id_joueurs(df):
    """
    Ajoute des colonnes 'Player1_ID' et 'Player2_ID' au DataFrame en extrayant les noms des joueurs à partir de 'match_id'.

    Parameters:
    df (pd.DataFrame): Le DataFrame contenant les données.

    Returns:
    pd.DataFrame: Le DataFrame mis à jour avec les colonnes des IDs des joueurs.
    """
    player1_ids = []
    player2_ids = []

    for index, row in df.iterrows():
        match_id = row['match_id']
        parts = match_id.split("-")
        player1 = parts[4]
        player2 = parts[5]
        player1_ids.append(player1)
        player2_ids.append(player2)

    df['Player1_ID'] = player1_ids
    df['Player2_ID'] = player2_ids

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





#pour détetrminer les neux joeurs d'un match
def get_match_players(df, row):
    """
    Determine the players of a tennis match based on the match ID.

    Parameters:
    row (pd.Series): A row from the DataFrame containing match data.

    Returns:
    tuple: A tuple containing the names of player1 and player2.

    """
    # ligne de la forme 20190219-M-Bergamo_CH-R32-Elliot_Benchetrit-Tallon_Griekspoor,49,0,0,5,2,15-15,8,True,2,4r38s3s3b2f2b3s3d@,,,2
    for index, row in df.iterrows():
        match_game = row['match_id']

        # On découpe l'ID.
        # Avec Elliot_Benchetrit-Tallon_Griekspoor, split("-") est plus simple
        parts = match_game.split("-")

        # Dans le format MCP standard :
        # parts[0]=date, [1]=sexe, [2]=tournoi, [3]=tour, [4]=joueur1, [5]=joueur2
        player1 = parts[4]
        player2 = parts[5]
        return (player1, player2)



# trouver le vainqueur du match
def get_match_winner(df, row):
    """
    Determine the winner of a tennis match based on the points won.

    Parameters:
    row (pd.Series): A row from the DataFrame containing match data.

    Returns:
    str: The winner of the match ("player1" or "player2").

    """

    match_game = row['match_id']
    df_match = pd.DataFrame(df[df['match_id'] == match_game])
    # le vainqueur du match est celui qui gagne le dernier point
    last_point_winner = df_match.iloc[-1]['PtWinner']
    joueurs = get_match_players(df, row)
    joueurs1 = joueurs[0]
    joueurs2 = joueurs[1]
    if last_point_winner == 1:
        return joueurs1
    else:
        return joueurs2


def ajouter_colonne_vainqueur(df):
    """
    Ajouter une colonne 'Winner' au DataFrame indiquant le vainqueur de chaque match.

    Parameters:
    df (pd.DataFrame): Le DataFrame contenant les données des matchs.

    Returns:
    pd.DataFrame: Le DataFrame mis à jour avec la colonne 'Winner'.
    """
    winners = []
    for index, row in df.iterrows():
        winner = get_match_winner(df, row)
        winners.append(winner)
    df['Winner'] = winners
    return df


def score_simple(score):

    """
    prends un score de la forme '15-30' et renvoie le score sous forme de tuple (a, b)
    a e b des entiers correspondant au nombre de points de chaque joueur dans le jeu en cours
    """
    points = score.split("-")
    point1 = points[0]
    point2 = points[1]
    score_dict = {"0": 0, "15": 1, "30": 2, "40": 3, "AD": 4}
    return (score_dict[point1], score_dict[point2])


def breakpoint(row):
    """
    Détermine si le point en cours est une opportunité de break pour le joueur qui reçoit.

    Parameters:
    row (pd.Series): Une ligne du DataFrame contenant les données du point.

    Returns:
    bool: True si c'est une opportunité de break, False sinon.
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





# Fonction pour attribuer les points
def get_points(df, tournament_type, round_reached, player):
    """Retourne les points ATP en fonction du tournoi et du round"""
    if round_reached == 'F' and df['Winner'] == player:
        return POINTS_BY_ROUND[tournament_type]['Winner']
    return POINTS_BY_ROUND[tournament_type][round_reached]


def enrich_match_context(df):
    """
    ajoute au DataFrame des informations sur le contexte du match
    telles que la surface, le niveau du tournoi et les points attribués pour le round
    Parameters:
    match_id (str): L'ID du match.
    df (pd.DataFrame): Le DataFrame contenant les données des matchs.
    Returns:
    pd.DataFrame: Le DataFrame mis à jour avec les informations de contexte du match.
    """
    for index, row in df.iterrows():
        match_id = row['match_id']
        # Extraire les informations du match_id
        parts = match_id.split("-")
        tournament_code = parts[2]
        # Récupérer les informations du tournoi
        tournament_info = TOURNAMENT_INFO.get(tournament_code, {})
        surface = tournament_info.get("surface", "Unknown")
        level = tournament_info.get("level", "Unknown")
        # Ajouter les informations au DataFrame
        df.at[index, 'Surface'] = surface
        df.at[index, 'Level'] = level
        # ajouter les points pour chaque joueur
        tournament_type = level
        round_code = parts[3]
        player1, player2 = get_match_players(df, row)
        points_player1 = get_points(df, tournament_type, round_code, player1)
        points_player2 = get_points(df, tournament_type, round_code, player2)
        df.at[index, 'Points_Player1'] = points_player1
        df.at[index, 'Points_Player2'] = points_player2
        df.at[index, 'Player1'] = player1
        df.at[index, 'Player2'] = player2
        # changer le format des points en score simple
        score = row['Pts']
        simple_score = score_simple(score)
        df.at[index, 'Simple_Score_Player1'] = simple_score[0]
        df.at[index, 'Simple_Score_Player2'] = simple_score[1]
        df.at[index, 'Is_Breakpoint'] = breakpoint(row)
        #ajout id joueurs
        df.at[index, 'Player1_ID'] = player1
        df.at[index, 'Player2_ID'] = player2

    return df_match


if __name__ == "__main__":
    test_file_path = "/Users/mathisverguet/live_win_probability_tennis/data/raw/processed/charting-m-points-2010s.csv"
    try:
        df = load_data(test_file_path)
        df_match = df[df['match_id'] == '20190219-M-Bergamo_CH-R32-Elliot_Benchetrit-Tallon_Griekspoor']
        print("Data loaded successfully:")
        print(df.head())
        #test get_match_players
        sample_row = df.iloc[0]
        players = get_match_players(df, sample_row)
        print(f"Players in the match: {players}")
        winner = get_match_winner(df, sample_row)
        print(f"Winner of the match: {winner}")
        ajouter_colonne_vainqueur(df_match)
        print("DataFrame with Winner column added:")
        print(df_match.head())
        print(score_simple('15-30'))
        bp = breakpoint(sample_row)
        print(f"Is it a breakpoint? {bp}")
        # test enrich_match_context
        df_match = enrich_match_context(df_match)
        print("DataFrame with match context enriched:")
        print(df_match.head())

    except Exception as e:
        print(e)
