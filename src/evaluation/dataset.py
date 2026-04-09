"""
T5 — Dataset d'évaluation ResearchPal v2.

15 paires question-réponse de référence :
  - 10 paires issues du corpus (Washington Capitals)
  - 3 paires adversariales (hors corpus, ambiguës, pièges à hallucination)
  - 2 paires multi-hop (nécessitent de combiner deux passages distincts)

Format : liste de dicts avec les clés :
  - question        : question de l'utilisateur
  - reference       : réponse de référence attendue
  - context_hint    : indice sur le passage source (pour le retrieval)
  - category        : "corpus" | "adversarial" | "multi_hop"
"""

EVAL_DATASET = [
    # -----------------------------------------------------------------------
    # CATÉGORIE : corpus (10 paires)
    # -----------------------------------------------------------------------
    {
        "id": "C01",
        "category": "corpus",
        "question": "Combien de buts Alexander Ovechkin a-t-il marqués lors de la saison 2024-25 ?",
        "reference": "Alexander Ovechkin a marqué 22 buts lors de la saison 2024-25 des Washington Capitals.",
        "context_hint": "Ovechkin statistiques 2024-25",
    },
    {
        "id": "C02",
        "category": "corpus",
        "question": "Qui est l'entraîneur-chef des Washington Capitals en 2025 ?",
        "reference": "Spencer Carbery est l'entraîneur-chef des Washington Capitals.",
        "context_hint": "entraîneur Washington Capitals 2025",
    },
    {
        "id": "C03",
        "category": "corpus",
        "question": "Quel est le nom de l'aréna où jouent les Washington Capitals ?",
        "reference": "Les Washington Capitals jouent à la Capital One Arena, située à Washington D.C.",
        "context_hint": "aréna Washington Capitals Capital One",
    },
    {
        "id": "C04",
        "category": "corpus",
        "question": "Quelle est la position d'Ovechkin sur la glace ?",
        "reference": "Alexander Ovechkin joue à l'aile gauche (left wing).",
        "context_hint": "Ovechkin position aile gauche",
    },
    {
        "id": "C05",
        "category": "corpus",
        "question": "En quelle année les Capitals ont-ils remporté la Coupe Stanley ?",
        "reference": "Les Washington Capitals ont remporté la Coupe Stanley en 2018.",
        "context_hint": "Capitals Coupe Stanley 2018",
    },
    {
        "id": "C06",
        "category": "corpus",
        "question": "Quel record d'Ovechkin a-t-il brisé concernant les buts en carrière en NHL ?",
        "reference": "Alexander Ovechkin a brisé le record de buts en carrière en NHL de Wayne Gretzky, qui était de 894 buts.",
        "context_hint": "Ovechkin record buts carrière Gretzky",
    },
    {
        "id": "C07",
        "category": "corpus",
        "question": "Quel gardien de but évolue pour les Capitals lors de la saison 2025-26 ?",
        "reference": "Logan Thompson est le gardien de but principal des Washington Capitals pour la saison 2025-26.",
        "context_hint": "gardien de but Capitals 2025-26",
    },
    {
        "id": "C08",
        "category": "corpus",
        "question": "Quel est le numéro de chandail d'Alexander Ovechkin ?",
        "reference": "Alexander Ovechkin porte le numéro 8 avec les Washington Capitals.",
        "context_hint": "Ovechkin numéro chandail 8",
    },
    {
        "id": "C09",
        "category": "corpus",
        "question": "Quelle est la nationalité d'Alexander Ovechkin ?",
        "reference": "Alexander Ovechkin est de nationalité russe. Il est né à Moscou, en Russie.",
        "context_hint": "Ovechkin nationalité russe Moscou",
    },
    {
        "id": "C10",
        "category": "corpus",
        "question": "Combien d'assistances Dylan Strome a-t-il réalisées lors de la saison 2024-25 ?",
        "reference": "Dylan Strome a enregistré des statistiques notables lors de la saison 2024-25 des Washington Capitals.",
        "context_hint": "Strome assistances statistiques 2024-25",
    },
    # -----------------------------------------------------------------------
    # CATÉGORIE : adversarial (3 paires — pièges à hallucination)
    # -----------------------------------------------------------------------
    {
        "id": "A01",
        "category": "adversarial",
        "question": "Quel est le score du match des Capitals d'hier soir ?",
        "reference": "Je ne dispose pas d'informations en temps réel sur les matchs récents. Veuillez consulter le site officiel de la NHL pour les scores actuels.",
        "context_hint": "score match récent hors corpus",
    },
    {
        "id": "A02",
        "category": "adversarial",
        "question": "Combien de buts Ovechkin a-t-il marqués lors de la finale de la Coupe du Monde 2026 ?",
        "reference": "Cette information n'est pas présente dans les documents indexés. Le hockey sur glace n'est pas un sport de la Coupe du Monde de football. La question semble confondre deux sports différents.",
        "context_hint": "question piège hors corpus Coupe du Monde football",
    },
    {
        "id": "A03",
        "category": "adversarial",
        "question": "Qui a remplacé Ovechkin comme meilleur buteur de la NHL après sa retraite ?",
        "reference": "Les documents indexés ne contiennent pas d'information sur la retraite d'Ovechkin ni sur son successeur. Selon les sources disponibles, Ovechkin est toujours actif.",
        "context_hint": "retraite Ovechkin successeur hors corpus",
    },
    # -----------------------------------------------------------------------
    # CATÉGORIE : multi_hop (2 paires — combiner 2 passages distincts)
    # -----------------------------------------------------------------------
    {
        "id": "M01",
        "category": "multi_hop",
        "question": "Combien de saisons Ovechkin a-t-il joué avec les Capitals depuis son record de buts, et quelle équipe les a draftés ?",
        "reference": "Alexander Ovechkin a été sélectionné en 1re position au repêchage de 2004 par les Washington Capitals. Depuis son record de 895 buts, il a continué à jouer avec les Capitals, ayant passé toute sa carrière dans cette organisation.",
        "context_hint": "Ovechkin draft 2004 Capitals record buts carrière saisons",
    },
    {
        "id": "M02",
        "category": "multi_hop",
        "question": "Quel est le total de points (buts + assistances) du meilleur buteur des Capitals lors de la saison 2024-25, et combien de matchs a-t-il joués ?",
        "reference": "Pour calculer le total de points d'un joueur, il faut combiner ses buts et ses assistances pour la saison 2024-25 et vérifier le nombre de matchs disputés selon les statistiques disponibles dans le corpus.",
        "context_hint": "meilleur buteur Capitals points buts assistances matchs 2024-25",
    },
]


def get_dataset_by_category(category: str = None):
    """Filtre le dataset par catégorie."""
    if category is None:
        return EVAL_DATASET
    return [e for e in EVAL_DATASET if e["category"] == category]


def print_dataset_summary():
    """Affiche un résumé du dataset."""
    from collections import Counter
    cats = Counter(e["category"] for e in EVAL_DATASET)
    print(f"Dataset d'évaluation : {len(EVAL_DATASET)} paires")
    for cat, count in cats.items():
        print(f"  - {cat}: {count} paires")
