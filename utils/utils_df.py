def selectionner_match_année(df, année):
    """Sélectionne les matchs supérieurs à une année donnée."""
    return df[df['year'] > année].reset_index(drop=True)


def selectionner_par_match(df, match_id):
    """Regroupe les données par match_id."""
    return df[df['match_id'] == match_id].sort_values('Pt').reset_index(drop=True)
