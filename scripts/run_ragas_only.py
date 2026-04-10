"""
run_ragas_only.py
=================
Relance uniquement l'évaluation RAGAS sans réexécuter le pipeline sur les 15 paires.
Utile quand collect_pipeline_outputs() a déjà tourné et que seul RAGAS a échoué.

Lancer depuis la racine du projet :
    python scripts/run_ragas_only.py
"""
import os
import sys
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from src.evaluation.ragas_eval import collect_pipeline_outputs, run_ragas_evaluation

print("📦 Collecte des outputs du pipeline...")
q, a, c, gt = collect_pipeline_outputs()

print("\n🔬 Lancement de l'évaluation RAGAS...")
run_ragas_evaluation(q, a, c, gt)