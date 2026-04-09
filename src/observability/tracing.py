from __future__ import annotations

import functools
import time
from contextlib import contextmanager
from typing import Any, Callable, Dict, Optional

# Initialisation locale Phoenix

_phoenix_session = None
_tracer = None


def setup_phoenix() -> bool:
    """
    Démarre Phoenix en mode local et configure l'instrumentation OpenTelemetry.

    Retourne True si Phoenix est disponible, False sinon (mode dégradé sans tracing).
    L'interface web est accessible sur http://localhost:6006.
    """
    global _phoenix_session, _tracer

    try:
        import phoenix as px
        from openinference.instrumentation.langchain import LangChainInstrumentor
        from opentelemetry import trace
        from opentelemetry.sdk.trace import TracerProvider
        from opentelemetry.sdk.trace.export import SimpleSpanProcessor
        from phoenix.otel import register

        # Démarrer Phoenix localement
        _phoenix_session = px.launch_app()
        print(f"✅ Phoenix démarré : {_phoenix_session.url}")

        # Configurer l'exporteur OTLP vers Phoenix
        tracer_provider = register(
            project_name="researchpal-v2",
            endpoint="http://localhost:6006/v1/traces",
        )

        # Instrumenter LangChain automatiquement
        LangChainInstrumentor().instrument(tracer_provider=tracer_provider)

        _tracer = tracer_provider.get_tracer("researchpal")
        return True

    except ImportError as e:
        print(f"⚠️  Phoenix non disponible ({e}) — tracing désactivé")
        return False
    except Exception as e:
        print(f"⚠️  Erreur Phoenix ({e}) — tracing désactivé")
        return False


def get_tracer():
    """Retourne le tracer OpenTelemetry (ou None si Phoenix non configuré)."""
    return _tracer


# Décorateurs et context managers pour les nœuds LangGraph 

@contextmanager
def node_span(name: str, attributes: Optional[Dict[str, Any]] = None):
    """
    Gestionnaire de contexte créant un span nommé pour un nœud LangGraph.

    Usage :
        with node_span("retrieve", {"retry_count": state["retry_count"]}):
            docs = vectorstore.similarity_search(query)

    Si Phoenix n'est pas configuré, agit comme un no-op transparent.
    """
    if _tracer is None:
        yield
        return

    with _tracer.start_as_current_span(name) as span:
        if attributes:
            for k, v in attributes.items():
                span.set_attribute(k, str(v))
        start = time.perf_counter()
        try:
            yield span
        finally:
            elapsed_ms = (time.perf_counter() - start) * 1000
            span.set_attribute("node.latency_ms", f"{elapsed_ms:.1f}")


def instrument_node(node_name: str):
    """
    Décorateur ajoutant automatiquement un span Phoenix à un nœud LangGraph.

    Usage :
        @instrument_node("retrieve")
        def retrieve_node(state: AgentState) -> AgentState: ...
    """
    def decorator(fn: Callable) -> Callable:
        @functools.wraps(fn)
        def wrapper(state: Dict[str, Any], *args, **kwargs) -> Dict[str, Any]:
            attrs = {
                "node.name": node_name,
                "retry_count": state.get("retry_count", 0),
                "tool_used": state.get("tool_used", ""),
            }
            with node_span(node_name, attrs):
                return fn(state, *args, **kwargs)
        return wrapper
    return decorator



# logs de session pour le rapport
def log_session_summary(session_id: str, queries: list, total_ms: float) -> None:
    """Affiche un résumé de session pour le rapport (traces/ dossier)."""
    print(f"\n📊 SESSION {session_id}")
    print(f"   Requêtes traitées : {len(queries)}")
    print(f"   Latence totale    : {total_ms:.0f} ms")
    for i, q in enumerate(queries, 1):
        print(f"   [{i}] {q[:80]}...")
    print(f"   → Traces disponibles sur http://localhost:6006\n")
