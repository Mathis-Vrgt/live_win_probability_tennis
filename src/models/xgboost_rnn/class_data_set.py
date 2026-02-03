# création de la classe DataSet pour pouvoir entrainer le modèle pytorch
import pandas as pd
import torch
from torch.utils.data import Dataset
class TennisDataset(Dataset):
    def __init__(self, data, feature_cols, target_col):
        self.X = data[feature_cols].values
        self.y = data[target_col].values
        self.data = data

    def __len__(self):
        return len(self.y)

    def players_to_index(self, data):
        players = pd.concat([data['Player1_Name'], data['Player2_Name']]).unique() # obtenir la liste unique des joueurs
        self.player_to_idx = {player: idx for idx, player in enumerate(players)}
        self.idx_to_player = {idx: player for player, idx in self.player_to_idx.items()}

    def surfaces_to_index(self):
        data = self.data
        surfaces = data['Surface'].unique()
        self.surface_to_idx = {surface: idx for idx, surface in enumerate(surfaces)}
        self.idx_to_surface = {idx: surface for surface, idx in self.surface_to_idx.items()}

    def tournaments_to_index(self):
        data = self.data
        tournaments = data['tournament_name'].unique()
        self.tournament_to_idx = {tournament: idx for idx, tournament in enumerate(tournaments)}
        self.idx_to_tournament = {idx: tournament for tournament, idx in self.tournament_to_idx.items()}

    def sequence_rnn(self, sequence_length=10):
        data = self.data.sort_values(['match_id', 'Pt']).reset_index(drop=True)
        all_sequences = []

        # Grouper par match pour éviter de recalculer data['match_id'] == matchs à chaque fois (plus rapide)
        for _, match_data in data.groupby('match_id'):
            winners = match_data['PtWinner'].values # On suppose 1 ou 2 ici

            for i in range(len(winners)):
                # On prend les points avant l'index i
                # Si i=0, la liste est vide. Si i=5, on prend de 0 à 4.
                current_seq = winners[max(0, i - sequence_length) : i].tolist()

                # Padding : on ajoute des 0 à gauche pour atteindre sequence_length
                padded_seq = [0] * (sequence_length - len(current_seq)) + current_seq
                all_sequences.append(padded_seq)

        self.sequences = torch.LongTensor(all_sequences)
        # j'ai l'ensemble des séquences pour chaque point dans l'ordre du dataframe sous forme de tenseur


    def __getitem__(self, idx):
        # Récupérer la séquence pré-calculée pour ce point précis
        seq = self.sequences[idx] # Résultat: un tenseur de taille (10,)

        # 2. Récupérer les indices des joueurs et décor (exemples)
        row = self.data.iloc[idx]
        p1 = self.player_to_idx.get(row['Player1_Name'], 0) # 0 si inconnu
        p2 = self.player_to_idx.get(row['Player2_Name'], 0)

        # 3. Récupérer la cible (Winner_Binary)
        target = self.y[idx]

        # On renvoie un dictionnaire (plus propre pour s'y retrouver après)
        return {
            'p1': torch.tensor(p1, dtype=torch.long),
            'p2': torch.tensor(p2, dtype=torch.long),
            'sequence': seq,
            'target': torch.tensor(target, dtype=torch.float)
        }
