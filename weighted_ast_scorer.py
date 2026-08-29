"""Scoring par Théorie des Ensembles Pondérés avec Attention Statique AST.

Module autonome, non branché au pipeline existant (retriever.py) — pour
tester la formule avant intégration éventuelle. Vrai Jaccard pondéré
(intersection / union) : IDF précalculé x score d'attention AST (déjà
fourni par l'appelant, pas recalculé ici) x poids fixe selon le type de
symbole (variable de portée ou import) côté requête ; IDF seul côté
symboles présents uniquement dans le chunk.
"""


def weighted_ast_attention_score(
    query_vars: dict[str, float],
    query_imports: set[str],
    chunk_symbols: set[str],
    chunk_imports: set[str],
    doc_weights: dict[str, float],
    var_weight: float = 2.0,
    import_weight: float = 2.5,
) -> float:
    """Jaccard pondéré entre une requête et un chunk candidat.

    query_vars : {nom_variable: attention_score}, attention_score déjà
        calculé en amont comme exp(-lambda * distance_ast) (proximité dans
        l'arbre AST par rapport au curseur) — cette fonction ne fait que le
        consommer, pas le recalculer.
    query_imports : imports actifs au niveau du curseur.
    chunk_symbols : symboles AST du chunk candidat (comparés à query_vars).
    chunk_imports : imports du chunk candidat (comparés à query_imports).
    doc_weights : poids IDF précalculés par symbole ; 1.0 si absent.

    Numérateur (intersection) : pour chaque symbole de la requête (variable
    ou import) aussi présent dans le chunk,
        poids = doc_weights.get(symbole, 1.0) * multiplicateur_de_type
    (le multiplicateur inclut le score d'attention pour les variables — les
    imports n'ont pas de notion de distance AST, donc pas d'attention_score,
    seulement leur propre multiplicateur `import_weight`).

    Dénominateur (union) : la somme ci-dessus pour TOUS les symboles de la
    requête (matchés ou non) + la somme des poids des symboles du chunk qui
    ne sont PAS dans la requête, où pour ceux-ci poids = doc_weights.get(v,
    1.0) tel quel (pas de multiplicateur de type, pas d'attention — lecture
    littérale de la spécification : seul le côté requête a un
    multiplicateur de type explicite).

    Toujours entre 0.0 et 1.0 (l'intersection est une somme partielle des
    termes déjà comptés côté requête dans l'union — jamais de terme compté
    en trop). Retourne 0.0 si requête et chunk sont tous les deux vides.
    """
    intersection_weight = 0.0
    union_weight = 0.0

    for var, attention_score in query_vars.items():
        weight = doc_weights.get(var, 1.0) * attention_score * var_weight
        union_weight += weight
        if var in chunk_symbols:
            intersection_weight += weight

    for imp in query_imports:
        weight = doc_weights.get(imp, 1.0) * import_weight
        union_weight += weight
        if imp in chunk_imports:
            intersection_weight += weight

    for symbol in chunk_symbols:
        if symbol not in query_vars:
            union_weight += doc_weights.get(symbol, 1.0)

    for imp in chunk_imports:
        if imp not in query_imports:
            union_weight += doc_weights.get(imp, 1.0)

    if union_weight == 0.0:
        return 0.0

    return intersection_weight / union_weight
