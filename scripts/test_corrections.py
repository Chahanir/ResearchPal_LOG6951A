"""
test_corrections.py
===================
Tests ciblés pour valider les deux corrections apportées à graph.py.

  - Test T3 : add_episodic_entry() est bien appelé après une résolution réussie
  - Test T4 : @instrument_node est bien appliqué sur chaque nœud du graphe

Lancer depuis la racine du projet :
    python scripts/test_corrections.py
"""

import os
import json
import sys
import tempfile

# Ajouter la racine du projet au path (même pattern que les autres scripts)
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import time
from pathlib import Path
from typing import Any, Dict
from unittest.mock import MagicMock, patch

# ── Couleurs terminal ──────────────────────────────────────────────────────────
GREEN  = "\033[92m"
RED    = "\033[91m"
YELLOW = "\033[93m"
RESET  = "\033[0m"
BOLD   = "\033[1m"

def ok(msg):  print(f"  {GREEN}✅ {msg}{RESET}")
def fail(msg): print(f"  {RED}❌ {msg}{RESET}")
def info(msg): print(f"  {YELLOW}ℹ  {msg}{RESET}")
def section(title): print(f"\n{BOLD}{'─'*55}\n  {title}\n{'─'*55}{RESET}")


# ══════════════════════════════════════════════════════════════════════════════
# TEST T3 — Mémoire épisodique : add_episodic_entry() appelée automatiquement
# ══════════════════════════════════════════════════════════════════════════════

def test_t3_episodic_memory_called():
    """
    Vérifie que run_agent() déclenche add_episodic_entry() quand la réponse
    est valide (grade=relevant, génération non vide et sans erreur).

    Stratégie : on mock graph.invoke() pour retourner un état synthétique,
    et on intercepte add_episodic_entry pour vérifier qu'elle est appelée.
    """
    section("T3 — Mémoire épisodique (add_episodic_entry)")

    calls = []  # Capture les appels à add_episodic_entry

    fake_state = {
        "question":           "Qui est l'entraîneur des Capitals ?",
        "generation":         "Spencer Carbery est l'entraîneur des Washington Capitals.",
        "documents":          [],
        "tool_used":          "corpus",
        "retry_count":        0,
        "grade":              "relevant",   # ← condition de sauvegarde
        "rewritten_question": None,
        "web_results":        None,
        "session_id":         "test",
        "episodic_context":   None,
        "messages":           [],
    }

    # On mock uniquement graph.invoke et add_episodic_entry
    with patch("src.agent.graph.add_episodic_entry", side_effect=lambda **kw: calls.append(kw)):
        with patch("src.agent.graph.build_graph") as mock_build:
            mock_graph = MagicMock()
            mock_graph.invoke.return_value = fake_state
            mock_build.return_value = mock_graph

            from src.agent.graph import run_agent
            run_agent("Qui est l'entraîneur des Capitals ?", thread_id="test", graph=mock_graph)

    if calls:
        ok(f"add_episodic_entry appelée — question : \"{calls[0].get('question', '')[:50]}\"")
        ok(f"tool_used transmis : {calls[0].get('tool_used')}")
        ok(f"quality_score transmis : {calls[0].get('quality_score')}")
    else:
        fail("add_episodic_entry N'A PAS été appelée — vérifier la condition dans run_agent()")
        return False

    # Cas négatif : grade=not_relevant ne doit PAS déclencher la sauvegarde
    calls.clear()
    fake_state_fail = {**fake_state, "grade": "not_relevant"}
    mock_graph.invoke.return_value = fake_state_fail  # ← appliquer le nouvel état au mock
    with patch("src.agent.graph.add_episodic_entry", side_effect=lambda **kw: calls.append(kw)):
        from src.agent.graph import run_agent
        run_agent("Question sans réponse", thread_id="test2", graph=mock_graph)

    mock_graph.invoke.return_value = fake_state_fail
    if not calls:
        ok("Cas négatif : grade=not_relevant → add_episodic_entry NON appelée (correct)")
    else:
        fail("Cas négatif : add_episodic_entry appelée même avec grade=not_relevant")

    return True


def test_t3_episodic_file_written():
    """
    Vérifie que add_episodic_entry() écrit réellement dans le fichier JSON
    et qu'il est lisible entre deux sessions (persistance).
    """
    section("T3 — Persistance fichier JSON épisodique")

    with tempfile.TemporaryDirectory() as tmpdir:
        episodic_file = Path(tmpdir) / "episodic_memory.json"

        with patch("src.agent.memory.EPISODIC_FILE", episodic_file):
            from src.agent.memory import add_episodic_entry, load_episodic_memory

            add_episodic_entry(
                question="Combien de buts Ovechkin a-t-il marqués ?",
                answer="44 buts lors de la saison 2024-25.",
                tool_used="corpus",
                quality_score=0.95,
            )

            if not episodic_file.exists():
                fail("Le fichier JSON n'a pas été créé")
                return False
            ok("Fichier JSON créé sur disque")

            entries = load_episodic_memory()
            if len(entries) == 1:
                ok(f"1 entrée persistée — question : \"{entries[0]['question'][:50]}\"")
                ok(f"quality_score : {entries[0]['quality_score']}")
            else:
                fail(f"Attendu 1 entrée, obtenu {len(entries)}")
                return False

            # Simuler un redémarrage : recharger depuis le fichier
            entries_after_restart = load_episodic_memory()
            if entries_after_restart:
                ok("Rechargement après redémarrage simulé : entrée retrouvée ✓")
            else:
                fail("Entrée perdue après rechargement — problème de persistance")

    return True


# ══════════════════════════════════════════════════════════════════════════════
# TEST T4 — Spans nommés : @instrument_node appliqué sur les nœuds
# ══════════════════════════════════════════════════════════════════════════════

EXPECTED_NODES = [
    "route_question",
    "execute_tools",
    "retrieve",
    "grade_documents",
    "rewrite_query",
    "generate",
]

def test_t4_decorators_present():
    """
    Vérifie que chaque fonction-nœud de graph.py est bien enveloppée par
    @instrument_node en inspectant leur __wrapped__ (functools.wraps le pose)
    et leur __name__ (wraps préserve le nom original).
    """
    section("T4 — Décorateurs @instrument_node présents sur les nœuds")

    # Import différé pour éviter les effets de bord au chargement
    import src.agent.graph as g

    node_funcs = {
        "route_question":  g.route_question_node,
        "execute_tools":   g.execute_tools_node,
        "retrieve":        g.retrieve_node,
        "grade_documents": g.grade_documents_node,
        "rewrite_query":   g.rewrite_query_node,
        "generate":        g.generate_node,
    }

    all_ok = True
    for name, fn in node_funcs.items():
        has_wrapped = hasattr(fn, "__wrapped__")
        name_preserved = fn.__name__ == f"{name}_node"
        if has_wrapped and name_preserved:
            ok(f"{name}_node → décoré (__wrapped__ présent, __name__ préservé)")
        else:
            fail(f"{name}_node → décorateur MANQUANT ou mal appliqué")
            info(f"  __wrapped__ : {has_wrapped} | __name__ : {fn.__name__}")
            all_ok = False

    return all_ok


def test_t4_span_fired():
    """
    Vérifie que node_span est bien appelé lors de l'exécution d'un nœud décoré,
    même sans Phoenix installé (le no-op de node_span doit se déclencher).
    """
    section("T4 — Span déclenché à l'exécution (sans Phoenix requis)")

    span_calls = []

    # Patch node_span pour capturer les appels
    from contextlib import contextmanager

    @contextmanager
    def fake_node_span(name, attributes=None):
        span_calls.append(name)
        yield

    with patch("src.observability.tracing.node_span", fake_node_span):
        # Créer un nœud factice décoré
        from src.observability.tracing import instrument_node

        @instrument_node("test_node")
        def dummy_node(state: Dict[str, Any]) -> Dict[str, Any]:
            return state

        dummy_node({"retry_count": 0, "tool_used": "corpus"})

    if "test_node" in span_calls:
        ok("node_span appelée avec le bon nom lors de l'exécution du nœud")
    else:
        fail("node_span N'A PAS été appelée — le décorateur ne fonctionne pas")
        return False

    return True


# ══════════════════════════════════════════════════════════════════════════════
# RUNNER
# ══════════════════════════════════════════════════════════════════════════════

def run_all():
    print(f"\n{BOLD}{'═'*55}")
    print("  test_corrections.py — Validation des corrections TP2")
    print(f"{'═'*55}{RESET}")

    results = {}

    # T3
    try:
        results["T3 — add_episodic_entry appelée"]  = test_t3_episodic_memory_called()
    except Exception as e:
        fail(f"Exception : {e}")
        results["T3 — add_episodic_entry appelée"] = False

    try:
        results["T3 — Persistance JSON"]            = test_t3_episodic_file_written()
    except Exception as e:
        fail(f"Exception : {e}")
        results["T3 — Persistance JSON"] = False

    # T4
    try:
        results["T4 — Décorateurs présents"]        = test_t4_decorators_present()
    except Exception as e:
        fail(f"Exception : {e}")
        results["T4 — Décorateurs présents"] = False

    try:
        results["T4 — Span déclenché"]              = test_t4_span_fired()
    except Exception as e:
        fail(f"Exception : {e}")
        results["T4 — Span déclenché"] = False

    # Récap
    section("Récapitulatif")
    passed = sum(1 for v in results.values() if v)
    total  = len(results)
    for label, result in results.items():
        status = f"{GREEN}PASS{RESET}" if result else f"{RED}FAIL{RESET}"
        print(f"  [{status}] {label}")

    print(f"\n  {BOLD}{passed}/{total} tests passés{RESET}\n")
    sys.exit(0 if passed == total else 1)


if __name__ == "__main__":
    run_all()