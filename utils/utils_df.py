import numpy as np
import pandas as pd

def selectionner_match_année(df, année):
    """Sélectionne les matchs supérieurs à une année donnée."""
    return df[df['year'] > année].reset_index(drop=True)


def selectionner_par_match(df, match_id):
    """Regroupe les données par match_id."""
    return df[df['match_id'] == match_id].sort_values('Pt').reset_index(drop=True)


def selectionner_match_fort_leverage(probabilite):
    # Calcule la différence entre chaque point et le précédent
    # np.diff calcule proba[i] - proba[i-1]
    mouvement = np.abs(np.diff(probabilite[:, 0]))

    # On prend les 20 moments où la proba a le plus sauté
    indices_leverage = np.argsort(mouvement)[-6:]

    return indices_leverage +1
