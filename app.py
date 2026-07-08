"""Dashboard en espanol del sistema Mundial 2026."""

from __future__ import annotations

import dataclasses
import json
import os
import subprocess
import sys
from collections import Counter
from datetime import datetime, timezone
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent
SRC_DIR = PROJECT_ROOT / "src"
if str(SRC_DIR) not in sys.path:
    sys.path.insert(0, str(SRC_DIR))

import pandas as pd
import plotly.express as px
import streamlit as st

from mundial.config import ARTIFACTS_DIR, load_groups
from mundial.history import load_world_cup_titles
from mundial.inference import load_predictor
from mundial.polymarket import filter_pending_markets, load_snapshot
from mundial.simulation import GROUPS, TournamentSimulator, validate_groups
from mundial.statistical import align_score_matrix
from mundial.tournament_state import (
    CANDIDATE_PATH,
    FIFA_SOURCE_URL,
    STATE_PATH,
    approve_candidate,
    fetch_fifa_state,
    load_tournament_state,
    utc_now,
    validate_state,
    write_state,
)

st.set_page_config(page_title="Inteligencia Mundial 2026", page_icon="⚽", layout="wide")

st.markdown(
    """
    <style>
    .block-container {
        padding-top: 1.35rem;
        padding-bottom: 2.5rem;
    }
    h1, h2, h3 {
        letter-spacing: 0;
    }
    .stMetric {
        background: linear-gradient(180deg, rgba(255, 255, 255, 0.055), rgba(255, 255, 255, 0.025));
        border: 1px solid rgba(255, 255, 255, 0.10);
        border-radius: 8px;
        padding: 0.7rem 0.8rem;
        box-shadow: 0 10px 24px rgba(0, 0, 0, 0.16);
    }
    div[data-testid="stDataFrame"] {
        border: 1px solid rgba(255, 255, 255, 0.08);
        border-radius: 8px;
        overflow: hidden;
    }
    .stTabs [data-baseweb="tab-list"] {
        gap: 0.25rem;
    }
    .stTabs [data-baseweb="tab"] {
        border-radius: 8px 8px 0 0;
        padding: 0.75rem 1rem;
    }
    div[data-testid="stExpander"] {
        border-radius: 8px;
        border-color: rgba(255, 255, 255, 0.10);
    }
    .wc-hero {
        border: 1px solid rgba(255, 255, 255, 0.10);
        border-radius: 8px;
        padding: 1.05rem 1.15rem;
        margin: 0 0 1rem 0;
        background:
            linear-gradient(135deg, rgba(12, 109, 86, 0.28), rgba(20, 27, 43, 0.72)),
            linear-gradient(90deg, rgba(255, 255, 255, 0.07), rgba(255, 255, 255, 0.01));
    }
    .wc-hero-title {
        font-size: clamp(1.55rem, 2.2vw, 2.25rem);
        line-height: 1.08;
        font-weight: 750;
        margin: 0 0 0.35rem 0;
    }
    .wc-hero-subtitle {
        color: rgba(255, 255, 255, 0.72);
        font-size: 0.98rem;
        margin: 0 0 0.8rem 0;
    }
    .wc-chip-row {
        display: flex;
        flex-wrap: wrap;
        gap: 0.45rem;
    }
    .wc-chip {
        border: 1px solid rgba(255, 255, 255, 0.12);
        border-radius: 999px;
        padding: 0.28rem 0.58rem;
        background: rgba(255, 255, 255, 0.065);
        color: rgba(255, 255, 255, 0.88);
        font-size: 0.82rem;
        white-space: nowrap;
    }
    .wc-match-card {
        border: 1px solid rgba(255, 255, 255, 0.10);
        border-radius: 8px;
        padding: 0.78rem 0.86rem;
        margin-bottom: 0.72rem;
        background: rgba(255, 255, 255, 0.032);
    }
    .wc-match-id {
        color: rgba(255, 255, 255, 0.58);
        font-size: 0.78rem;
        margin-bottom: 0.2rem;
    }
    .wc-match-title {
        font-size: 1rem;
        font-weight: 700;
        margin-bottom: 0.38rem;
    }
    .wc-match-meta {
        color: rgba(255, 255, 255, 0.76);
        font-size: 0.86rem;
        line-height: 1.45;
    }
    .model-card {
        border: 1px solid rgba(255, 255, 255, 0.10);
        border-radius: 8px;
        padding: 0.82rem 0.9rem;
        background: rgba(255, 255, 255, 0.035);
        min-height: 112px;
    }
    .model-card-title {
        color: rgba(255, 255, 255, 0.62);
        font-size: 0.78rem;
        margin-bottom: 0.35rem;
    }
    .model-card-status {
        font-size: 1.35rem;
        font-weight: 750;
        margin-bottom: 0.3rem;
    }
    .model-card-detail {
        color: rgba(255, 255, 255, 0.74);
        font-size: 0.84rem;
        line-height: 1.35;
    }
    .model-ok {
        color: #2ec4a6;
    }
    .model-review {
        color: #f2c94c;
    }
    .market-panel {
        border: 1px solid rgba(242, 201, 76, 0.28);
        border-radius: 8px;
        padding: 0.9rem 0.95rem;
        background:
            linear-gradient(135deg, rgba(242, 201, 76, 0.12), rgba(255, 255, 255, 0.02));
        margin-top: 0.85rem;
    }
    .market-panel-title {
        color: #f2c94c;
        font-weight: 720;
        margin-bottom: 0.18rem;
    }
    .market-panel-caption {
        color: rgba(255, 255, 255, 0.68);
        font-size: 0.84rem;
        margin-bottom: 0.65rem;
    }
    [data-testid="stSidebar"] {
        background: #202129;
    }
    [data-testid="stSidebar"] [data-testid="stVerticalBlock"] {
        gap: 0.8rem;
    }
    .sidebar-card {
        border: 1px solid rgba(255, 255, 255, 0.10);
        border-radius: 8px;
        padding: 0.85rem 0.9rem;
        background: rgba(255, 255, 255, 0.035);
        margin-bottom: 0.8rem;
    }
    .sidebar-title {
        font-size: 0.92rem;
        font-weight: 760;
        margin-bottom: 0.35rem;
    }
    .sidebar-line {
        color: rgba(255, 255, 255, 0.68);
        font-size: 0.8rem;
        line-height: 1.45;
    }
    .sidebar-hero {
        border: 1px solid rgba(46, 196, 166, 0.22);
        border-radius: 8px;
        padding: 0.9rem;
        background: linear-gradient(135deg, rgba(46, 196, 166, 0.14), rgba(255, 255, 255, 0.035));
        margin-bottom: 0.8rem;
    }
    .sidebar-kicker {
        color: #2ec4a6;
        font-size: 0.74rem;
        font-weight: 760;
        text-transform: uppercase;
        letter-spacing: 0.04em;
        margin-bottom: 0.28rem;
    }
    .sidebar-next {
        font-size: 0.98rem;
        font-weight: 760;
        line-height: 1.25;
        margin-bottom: 0.35rem;
    }
    .sidebar-pill-row {
        display: flex;
        flex-wrap: wrap;
        gap: 0.35rem;
        margin-top: 0.55rem;
    }
    .sidebar-pill {
        border-radius: 999px;
        padding: 0.22rem 0.5rem;
        background: rgba(255, 255, 255, 0.08);
        color: rgba(255, 255, 255, 0.82);
        font-size: 0.74rem;
    }
    </style>
    """,
    unsafe_allow_html=True,
)

FLAG_BY_TEAM = {
    "Algeria": "🇩🇿",
    "Argentina": "🇦🇷",
    "Australia": "🇦🇺",
    "Austria": "🇦🇹",
    "Belgium": "🇧🇪",
    "Bosnia and Herzegovina": "🇧🇦",
    "Brazil": "🇧🇷",
    "Cabo Verde": "🇨🇻",
    "Canada": "🇨🇦",
    "Colombia": "🇨🇴",
    "Congo DR": "🇨🇩",
    "Croatia": "🇭🇷",
    "Curaçao": "🇨🇼",
    "Czechia": "🇨🇿",
    "Côte d'Ivoire": "🇨🇮",
    "Ecuador": "🇪🇨",
    "Egypt": "🇪🇬",
    "England": "🏴",
    "France": "🇫🇷",
    "Germany": "🇩🇪",
    "Ghana": "🇬🇭",
    "Haiti": "🇭🇹",
    "Iran": "🇮🇷",
    "Iraq": "🇮🇶",
    "Japan": "🇯🇵",
    "Jordan": "🇯🇴",
    "Korea Republic": "🇰🇷",
    "Mexico": "🇲🇽",
    "Morocco": "🇲🇦",
    "Netherlands": "🇳🇱",
    "New Zealand": "🇳🇿",
    "Norway": "🇳🇴",
    "Panama": "🇵🇦",
    "Paraguay": "🇵🇾",
    "Portugal": "🇵🇹",
    "Qatar": "🇶🇦",
    "Saudi Arabia": "🇸🇦",
    "Scotland": "🏴",
    "Senegal": "🇸🇳",
    "South Africa": "🇿🇦",
    "Spain": "🇪🇸",
    "Sweden": "🇸🇪",
    "Switzerland": "🇨🇭",
    "Tunisia": "🇹🇳",
    "Türkiye": "🇹🇷",
    "United States": "🇺🇸",
    "Uruguay": "🇺🇾",
    "Uzbekistan": "🇺🇿",
}


def team_label(team: str | None) -> str:
    if not team:
        return "Por definir"
    flag = FLAG_BY_TEAM.get(team)
    return f"{flag} {team}" if flag else team


def format_kickoff(value: str) -> str:
    parsed = _parse_kickoff(value)
    if parsed == datetime.max.replace(tzinfo=timezone.utc):
        return value or "N/A"
    return parsed.astimezone(timezone.utc).strftime("%d/%m %H:%M UTC")


def next_match_label(state) -> tuple[str, str]:
    if not state:
        return "Sin calendario oficial", "Actualice FIFA para cargar estado"
    pending = sorted(
        [match for match in state.matches.values() if not match.is_finished],
        key=lambda match: _parse_kickoff(match.kickoff),
    )
    if not pending:
        return "Torneo cerrado", "No quedan partidos pendientes"
    match = pending[0]
    title = f"{match.match_id} · {team_label(match.team_a)} vs {team_label(match.team_b)}"
    subtitle = f"{_phase_label(match.phase)} · {format_kickoff(match.kickoff)}"
    return title, subtitle


def sidebar_model_summary() -> tuple[str, str, str]:
    metrics = _load_artifact_json(ARTIFACTS_DIR / "metrics.json")
    selected = str(metrics.get("selected_model") or "N/A")
    served = metrics.get("test_hybrid_served") if isinstance(metrics.get("test_hybrid_served"), dict) else {}
    ece = _format_number(served.get("ece")) if served else "N/A"
    accuracy = _format_percent(served.get("accuracy")) if served else "N/A"
    return selected, ece, accuracy


def sidebar_market_count() -> int:
    snapshot = _load_artifact_json(ARTIFACTS_DIR / "polymarket_snapshot.json")
    markets = snapshot.get("markets")
    return len(markets) if isinstance(markets, list) else 0


def render_app_header(model_mode: str, state, update_status: dict[str, object] | None) -> None:
    official_matches = len(state.finished_matches) if state else 0
    phase = "Sin estado oficial"
    if state:
        pending = sorted(
            [match for match in state.matches.values() if not match.is_finished],
            key=lambda match: _parse_kickoff(match.kickoff),
        )
        phase = _phase_label(pending[0].phase if pending else "final")
    updated = (state.approved_at or state.source_updated_at or state.generated_at) if state else "N/A"
    update_label = "FIFA OK" if not update_status or update_status.get("ok", True) else "FIFA con fallback"
    st.markdown(
        f"""
        <div class="wc-hero">
            <div class="wc-hero-title">Inteligencia deportiva — Mundial 2026</div>
            <div class="wc-hero-subtitle">Prediccion de partidos, simulacion Monte Carlo y validacion del modelo en un solo tablero.</div>
            <div class="wc-chip-row">
                <span class="wc-chip">⚙️ {model_mode}</span>
                <span class="wc-chip">✅ {official_matches}/104 oficiales</span>
                <span class="wc-chip">🏁 {phase}</span>
                <span class="wc-chip">🕒 {updated}</span>
                <span class="wc-chip">📡 {update_label}</span>
            </div>
        </div>
        """,
        unsafe_allow_html=True,
    )


def render_match_card(column, match, favorite: str, state_label: str, forced: str) -> None:
    column.markdown(
        f"""
        <div class="wc-match-card">
            <div class="wc-match-id">{match.match_id} · {state_label}{forced}</div>
            <div class="wc-match-title">{team_label(match.team_a)} vs {team_label(match.team_b)}</div>
            <div class="wc-match-meta">
                Favorito: <b>{team_label(favorite)}</b><br>
                Avanza {team_label(match.team_a)}: <b>{match.probability_a:.0%}</b> ·
                Avanza {team_label(match.team_b)}: <b>{match.probability_b:.0%}</b><br>
                90 min: {team_label(match.team_a)} {match.match_probability_a:.0%} ·
                Empate {match.draw_probability:.0%} ·
                {team_label(match.team_b)} {match.match_probability_b:.0%}<br>
                Ganador mostrado: <b>{team_label(match.winner)}</b>
            </div>
        </div>
        """,
        unsafe_allow_html=True,
    )


@st.cache_resource
def resources(state_version: int):
    predictor, mode = load_predictor()
    return predictor, TournamentSimulator(predictor), mode


@st.cache_data(show_spinner="Simulando el torneo...")
def run_tournament_simulation(
    _simulator,
    groups: dict[str, list[str]],
    overrides: dict[str, str],
    runs: int,
    seed: int,
    use_official_state: bool,
    state_version: int,
    _tournament_state,
):
    """Cachea el Monte Carlo para no recomputarlo en cada rerun.

    `_simulator` y `_tournament_state` se excluyen del hash por el prefijo `_`;
    `state_version` y `use_official_state` capturan cualquier cambio del estado
    oficial, de modo que la caché solo se invalida cuando cambian los grupos,
    los overrides, las corridas, la semilla o el estado aprobado.
    """
    return _simulator.simulate(
        groups,
        overrides=overrides,
        runs=runs,
        seed=seed,
        tournament_state=_tournament_state if use_official_state else None,
    )


def groups_from_editor(default_groups: dict[str, list[str]], *, disabled: bool = False) -> dict[str, list[str]]:
    all_teams = [team for group in GROUPS for team in default_groups[group]]
    edited: dict[str, list[str]] = {}
    for group in GROUPS:
        with st.expander(f"Grupo {group}", expanded=group in "ABC"):
            edited[group] = [
                st.selectbox(
                    f"Seleccion {position + 1}", all_teams,
                    index=all_teams.index(default_groups[group][position]),
                    key=f"group_{group}_{position}",
                    disabled=disabled,
                    format_func=team_label,
                )
                for position in range(4)
            ]
    return edited


def groups_from_session(default_groups: dict[str, list[str]], *, enabled: bool) -> dict[str, list[str]]:
    if not enabled:
        return {group: list(default_groups[group]) for group in GROUPS}
    return {
        group: [
            st.session_state.get(f"group_{group}_{position}", default_groups[group][position])
            for position in range(4)
        ]
        for group in GROUPS
    }


def exact_score_rows_from_matrix(
    matrix,
    team_a: str,
    team_b: str,
    limit: int = 6,
) -> list[dict[str, object]]:
    rows = []
    for goals_a, row in enumerate(matrix):
        for goals_b, probability in enumerate(row):
            rows.append(
                {
                    "Marcador": f"{team_a} {goals_a}-{goals_b} {team_b}",
                    "Probabilidad": float(probability),
                }
            )
    return sorted(rows, key=lambda item: item["Probabilidad"], reverse=True)[:limit]


def exact_score_rows(prediction, limit: int = 6) -> list[dict[str, object]]:
    if prediction.score_probabilities is None:
        return []
    return exact_score_rows_from_matrix(
        prediction.score_probabilities,
        prediction.team_a,
        prediction.team_b,
        limit,
    )


def format_exact_score_frame(rows: list[dict[str, object]], team_a: str, team_b: str) -> pd.DataFrame:
    frame = pd.DataFrame(rows)
    if frame.empty:
        return frame
    frame["Marcador"] = frame["Marcador"].str.replace(team_a, team_label(team_a), regex=False)
    frame["Marcador"] = frame["Marcador"].str.replace(team_b, team_label(team_b), regex=False)
    frame["Probabilidad"] = frame["Probabilidad"].map(lambda value: f"{value:.2%}")
    return frame


def render_probability_cards(prediction) -> None:
    columns = st.columns(3)
    columns[0].metric(f"Gana {team_label(prediction.team_a)}", f"{prediction.prob_a:.1%}")
    columns[1].metric("Empate", f"{prediction.prob_draw:.1%}")
    columns[2].metric(f"Gana {team_label(prediction.team_b)}", f"{prediction.prob_b:.1%}")
    st.subheader(f"Marcador exacto mas probable: {prediction.likely_score[0]}–{prediction.likely_score[1]}")
    exact_scores = exact_score_rows(prediction)
    if exact_scores:
        score_frame = format_exact_score_frame(exact_scores, prediction.team_a, prediction.team_b)
        with st.expander("Ver marcadores exactos mas probables", expanded=True):
            st.dataframe(score_frame, hide_index=True, width="stretch")
    bp = getattr(prediction, "base_probabilities", None)
    mp = getattr(prediction, "market_probabilities", None)
    mw = getattr(prediction, "market_weight", None)
    if bp is not None:
        if mp is not None and mw is not None and mw > 0:
            with st.expander("Detalle: modelo vs mercado"):
                col1, col2, col3 = st.columns(3)
                col1.caption("Modelo DL")
                col1.write(
                    f"{team_label(prediction.team_a)}: {bp[0]:.1%} | "
                    f"Empate: {bp[1]:.1%} | {team_label(prediction.team_b)}: {bp[2]:.1%}"
                )
                col2.caption("Polymarket")
                col2.write(
                    f"{team_label(prediction.team_a)}: {mp[0]:.1%} | "
                    f"Empate: {mp[1]:.1%} | {team_label(prediction.team_b)}: {mp[2]:.1%}"
                )
                col3.caption("Combinado")
                alpha_str = f"α={mw:.2f}" if mw is not None else "N/A"
                col3.write(f"Peso {alpha_str} | Capturado: {getattr(prediction, 'market_as_of', None) or 'N/A'}")
        else:
            st.info("Este cruce no usa una cotización Polymarket válida; se sirve el modelo DL.")


def render_market_opinion(prediction) -> None:
    team_a = prediction.team_a
    team_b = prediction.team_b
    market = _market_snapshot_for_match(team_a, team_b)
    if not market:
        st.caption("Mercado: sin cotización Polymarket vigente para este cruce.")
        return

    st.markdown(
        """
        <div class="market-panel">
            <div class="market-panel-title">Opinión actual del mercado</div>
            <div class="market-panel-caption">Polymarket · resultado en 90 minutos</div>
        </div>
        """,
        unsafe_allow_html=True,
    )
    col_a, col_draw, col_b = st.columns(3)
    col_a.metric(team_label(team_a), _format_percent(market.get("prob_a")))
    col_draw.metric("Empate", _format_percent(market.get("prob_draw")))
    col_b.metric(team_label(team_b), _format_percent(market.get("prob_b")))

    metadata = []
    if isinstance(market.get("total_liquidity"), (float, int)):
        metadata.append(f"liquidez ${market['total_liquidity']:,.0f}")
    if isinstance(market.get("spread"), (float, int)):
        metadata.append(f"spread {market['spread']:.1%}")
    if market.get("captured_at"):
        metadata.append(f"capturado {market['captured_at']}")
    if metadata:
        st.caption(" · ".join(metadata))

    if prediction.score_probabilities is not None:
        target = (market.get("prob_a"), market.get("prob_draw"), market.get("prob_b"))
        if all(isinstance(value, (float, int)) for value in target):
            market_matrix = align_score_matrix(prediction.score_probabilities, target)
            market_rows = exact_score_rows_from_matrix(market_matrix, team_a, team_b)
            market_frame = format_exact_score_frame(market_rows, team_a, team_b)
            with st.expander("Ver marcadores exactos implicitos por mercado", expanded=False):
                st.caption(
                    "Polymarket no entrega correct score en este snapshot; esta tabla conserva la forma de marcadores "
                    "del modelo y reemplaza solo la masa 1-X-2 por la probabilidad del mercado."
                )
                st.dataframe(market_frame, hide_index=True, width="stretch")


def _parse_kickoff(value: str) -> datetime:
    try:
        parsed = datetime.fromisoformat(value.replace("Z", "+00:00"))
    except ValueError:
        return datetime.max.replace(tzinfo=timezone.utc)
    return parsed if parsed.tzinfo else parsed.replace(tzinfo=timezone.utc)


def _phase_label(phase: str) -> str:
    return {
        "group": "Fase de grupos",
        "round_of_32": "Ronda de 32",
        "round_of_16": "Octavos",
        "quarter_final": "Cuartos",
        "semi_final": "Semifinales",
        "third_place": "Tercer puesto",
        "final": "Final",
    }.get(phase, phase or "N/A")


def _load_artifact_json(path: Path) -> dict[str, object]:
    if not path.exists():
        return {}
    try:
        payload = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, ValueError, json.JSONDecodeError):
        return {}
    return payload if isinstance(payload, dict) else {}


def _market_snapshot_for_match(team_a: str, team_b: str) -> dict[str, object] | None:
    snapshot = _load_artifact_json(ARTIFACTS_DIR / "polymarket_snapshot.json")
    markets = snapshot.get("markets")
    if not isinstance(markets, list):
        return None
    for market in markets:
        if not isinstance(market, dict):
            continue
        pair = {market.get("team_a"), market.get("team_b")}
        if pair == {team_a, team_b}:
            if market.get("team_a") == team_a:
                return market
            adjusted = dict(market)
            adjusted["team_a"] = team_a
            adjusted["team_b"] = team_b
            adjusted["prob_a"] = market.get("prob_b")
            adjusted["prob_b"] = market.get("prob_a")
            return adjusted
    return None


def _format_number(value: object, digits: int = 3) -> str:
    return f"{value:.{digits}f}" if isinstance(value, (float, int)) else "N/A"


def _format_percent(value: object) -> str:
    return f"{value:.1%}" if isinstance(value, (float, int)) else "N/A"


def style_plotly_figure(figure):
    figure.update_layout(
        template="plotly_dark",
        paper_bgcolor="rgba(0,0,0,0)",
        plot_bgcolor="rgba(0,0,0,0)",
        margin=dict(l=10, r=10, t=20, b=10),
        font=dict(size=13),
    )
    return figure


def _metric_status(value: object, threshold: float, *, lower_is_better: bool = True) -> str:
    if not isinstance(value, (float, int)):
        return "Revisar"
    return "OK" if (value <= threshold if lower_is_better else value >= threshold) else "Revisar"


def render_model_health_cards(checks: list[dict[str, str]]) -> None:
    columns = st.columns(len(checks))
    for column, check in zip(columns, checks, strict=True):
        status = check["Estado"]
        status_class = "model-ok" if status == "OK" else "model-review"
        column.markdown(
            f"""
            <div class="model-card">
                <div class="model-card-title">{check["ID"]}</div>
                <div class="model-card-status {status_class}">{status}</div>
                <div class="model-card-detail"><b>{check["Criterio"]}</b><br>{check["Evidencia"]}</div>
            </div>
            """,
            unsafe_allow_html=True,
        )


def model_insights(
    selected_model: object,
    model_metrics: dict[str, object],
    served_metrics: dict[str, object],
    metrics: dict[str, object],
) -> list[dict[str, str]]:
    insights: list[dict[str, str]] = []
    if selected_model and model_metrics.get("mean_log_loss") is not None:
        insights.append(
            {
                "Tipo": "Fortaleza",
                "Hallazgo": "Backbone seleccionado por CV temporal",
                "Evidencia": f"{selected_model} · log-loss {_format_number(model_metrics.get('mean_log_loss'))}",
            }
        )
    if isinstance(served_metrics.get("ece"), (float, int)):
        status = "Fortaleza" if served_metrics["ece"] <= 0.05 else "Revisar"
        insights.append(
            {
                "Tipo": status,
                "Hallazgo": "Calibracion probabilistica",
                "Evidencia": f"ECE {_format_number(served_metrics.get('ece'))}",
            }
        )
    if _has_no_predicted_draws(served_metrics):
        insights.append(
            {
                "Tipo": "Riesgo",
                "Hallazgo": "No predijo empates en el test bloqueado",
                "Evidencia": "Revisar sesgo de clase empate",
            }
        )
    if metrics.get("calibration_promoted") is False:
        insights.append(
            {
                "Tipo": "Decision",
                "Hallazgo": "Calibracion final no promovida",
                "Evidencia": "Se sirve la variante que no degrada el test",
            }
        )
    insights.append(
        {
            "Tipo": "Contexto",
            "Hallazgo": "Dixon-Coles distribuye marcadores",
            "Evidencia": "No reemplaza el backbone 1-X-2",
        }
    )
    return insights


def _model_comparison_rows(metrics: dict[str, object], selected_model: object) -> list[dict[str, object]]:
    models = metrics.get("models")
    if not isinstance(models, dict):
        return []
    rows = []
    for model_name, values in models.items():
        if not isinstance(values, dict):
            continue
        rows.append(
            {
                "Modelo": model_name,
                "Seleccionado": "Si" if model_name == selected_model else "",
                "Log-loss CV": values.get("mean_log_loss"),
                "Brier CV": values.get("mean_brier"),
                "Macro F1 CV": values.get("mean_macro_f1"),
            }
        )
    return sorted(rows, key=lambda row: row["Log-loss CV"] if isinstance(row["Log-loss CV"], (float, int)) else 99)


def _fold_rows(model_metrics: dict[str, object]) -> list[dict[str, object]]:
    folds = model_metrics.get("folds")
    if not isinstance(folds, list):
        return []
    rows = []
    for fold in folds:
        if not isinstance(fold, dict):
            continue
        rows.append(
            {
                "Fold": fold.get("fold"),
                "Accuracy": fold.get("accuracy"),
                "Macro F1": fold.get("macro_f1"),
                "Log-loss": fold.get("log_loss"),
                "Brier": fold.get("brier"),
                "ECE": fold.get("ece"),
            }
        )
    return rows


def _confusion_matrix(metrics_block: dict[str, object]) -> list[list[object]]:
    matrix = metrics_block.get("confusion_matrix")
    if not isinstance(matrix, list) or len(matrix) != 3:
        return []
    return matrix


def _confusion_frame(metrics_block: dict[str, object], *, percent: bool = False) -> pd.DataFrame:
    labels = ["Gana A", "Empate", "Gana B"]
    matrix = _confusion_matrix(metrics_block)
    if not matrix:
        return pd.DataFrame(columns=["Real", *labels])
    frame = pd.DataFrame(matrix, columns=labels).assign(Real=labels)[["Real", *labels]]
    if not percent:
        return frame
    normalized = frame.copy()
    row_totals = normalized[labels].sum(axis=1).replace(0, pd.NA)
    normalized[labels] = normalized[labels].div(row_totals, axis=0).fillna(0.0)
    for label in labels:
        normalized[label] = normalized[label].map(lambda value: f"{value:.1%}")
    return normalized


def _has_no_predicted_draws(metrics_block: dict[str, object]) -> bool:
    matrix = _confusion_matrix(metrics_block)
    return bool(matrix) and sum(row[1] for row in matrix if isinstance(row, list) and len(row) > 1) == 0


def confusion_heatmap(metrics_block: dict[str, object], *, percent: bool = False):
    labels = ["Gana A", "Empate", "Gana B"]
    matrix = _confusion_matrix(metrics_block)
    if not matrix:
        return None
    frame = pd.DataFrame(matrix, columns=labels, index=labels)
    z = frame.astype(float)
    text = frame.astype(str)
    colorbar_title = "Partidos"
    if percent:
        row_totals = z.sum(axis=1).replace(0, pd.NA)
        z = z.div(row_totals, axis=0).fillna(0.0)
        text = z.map(lambda value: f"{value:.1%}")
        colorbar_title = "%"
    figure = px.imshow(
        z,
        labels=dict(x="Prediccion", y="Real", color=colorbar_title),
        x=labels,
        y=labels,
        text_auto=False,
        color_continuous_scale="Teal",
        aspect="auto",
    )
    figure.update_traces(text=text.to_numpy(), texttemplate="%{text}", hovertemplate="Real=%{y}<br>Pred=%{x}<br>%{text}<extra></extra>")
    figure.update_xaxes(side="top")
    return style_plotly_figure(figure)


def render_validation_section(state, update_status: dict[str, object] | None) -> None:
    st.header("Validacion del modelo")

    metrics = _load_artifact_json(ARTIFACTS_DIR / "metrics.json")
    manifest = _load_artifact_json(ARTIFACTS_DIR / "artifact_manifest.json")
    selected_model = metrics.get("selected_model") or manifest.get("selected_model")
    model_metrics = {}
    if isinstance(metrics.get("models"), dict) and selected_model in metrics["models"]:
        raw_model_metrics = metrics["models"][selected_model]
        if isinstance(raw_model_metrics, dict):
            model_metrics = raw_model_metrics
    raw_metrics = metrics.get("test_raw") if isinstance(metrics.get("test_raw"), dict) else {}
    calibrated_metrics = metrics.get("test_calibrated") if isinstance(metrics.get("test_calibrated"), dict) else {}
    served_metrics = metrics.get("test_hybrid_served") if isinstance(metrics.get("test_hybrid_served"), dict) else {}
    diagnostics = metrics.get("diagnostics") if isinstance(metrics.get("diagnostics"), dict) else {}
    nuts = diagnostics.get("nuts_audit") if isinstance(diagnostics.get("nuts_audit"), dict) else {}
    dixon_production = (
        diagnostics.get("dixon_coles_production")
        if isinstance(diagnostics.get("dixon_coles_production"), dict)
        else {}
    )
    splits = metrics.get("splits") if isinstance(metrics.get("splits"), dict) else {}

    checks = [
        {
            "ID": "MODEL-001",
            "Criterio": "Seleccion temporal del backbone",
            "Estado": "OK" if selected_model and model_metrics else "Revisar",
            "Evidencia": (
                f"{selected_model or 'N/A'} seleccionado por {metrics.get('selection_metric', 'N/A')} · "
                f"log-loss CV {_format_number(model_metrics.get('mean_log_loss'))}"
            ),
            "Lectura": "Menor log-loss CV implica mejores probabilidades promedio en cortes temporales.",
        },
        {
            "ID": "MODEL-002",
            "Criterio": "Generalizacion en test bloqueado",
            "Estado": "OK" if served_metrics else "Revisar",
            "Evidencia": (
                f"Accuracy {_format_percent(served_metrics.get('accuracy'))} · "
                f"log-loss {_format_number(served_metrics.get('log_loss'))} · "
                f"Brier {_format_number(served_metrics.get('brier'))}"
            ),
            "Lectura": "Mide rendimiento fuera de muestra del predictor que se sirve en la app.",
        },
        {
            "ID": "MODEL-003",
            "Criterio": "Calibracion probabilistica",
            "Estado": _metric_status(served_metrics.get("ece"), 0.05),
            "Evidencia": (
                f"ECE servido {_format_number(served_metrics.get('ece'))} · "
                f"calibracion promovida: {'si' if metrics.get('calibration_promoted') else 'no'}"
            ),
            "Lectura": "ECE bajo indica que confianza y frecuencia observada estan alineadas.",
        },
        {
            "ID": "MODEL-004",
            "Criterio": "Modelo de goles",
            "Estado": "OK" if served_metrics else "Revisar",
            "Evidencia": (
                f"MAE goles A {_format_number(served_metrics.get('mae_goals_a'))} · "
                f"MAE goles B {_format_number(served_metrics.get('mae_goals_b'))}"
            ),
            "Lectura": "Evalua la parte de marcador que alimenta Dixon-Coles y score probable.",
        },
        {
            "ID": "MODEL-005",
            "Criterio": "Auditoria bayesiana",
            "Estado": "OK" if nuts.get("max_rhat") == 1.0 and isinstance(nuts.get("min_ess_bulk"), (float, int)) else "Revisar",
            "Evidencia": (
                f"max R-hat {_format_number(nuts.get('max_rhat'))} · "
                f"min ESS {_format_number(nuts.get('min_ess_bulk'), 0)} · "
                f"partidos auditados {_format_number(nuts.get('matches'), 0)}"
            ),
            "Lectura": "Controla estabilidad de la capa bayesiana usada para incertidumbre.",
        },
    ]

    col1, col2, col3, col4 = st.columns(4)
    col1.metric("Backbone", str(selected_model or "N/A"))
    col2.metric("Accuracy test", _format_percent(served_metrics.get("accuracy")))
    col3.metric("Log-loss test", _format_number(served_metrics.get("log_loss")))
    col4.metric("ECE test", _format_number(served_metrics.get("ece")))
    st.caption(
        f"Artefacto v{manifest.get('version', 'N/A')} · as-of {manifest.get('as_of_date', 'N/A')} · "
        f"{manifest.get('posterior_draws', 'N/A')} muestras posteriores · "
        f"produccion: {splits.get('production', 'N/A')} partidos"
    )

    st.subheader("Semaforo del modelo")
    render_model_health_cards(checks)

    st.subheader("Hallazgos")
    insight_frame = pd.DataFrame(model_insights(selected_model, model_metrics, served_metrics, metrics))
    st.dataframe(insight_frame, hide_index=True, width="stretch")

    with st.expander("Ver criterios completos"):
        st.dataframe(pd.DataFrame(checks), hide_index=True, width="stretch")

    st.subheader("Comparacion raw, calibrado y servido")
    metric_rows = [
        {
            "Corte": "Test raw",
            "Modelo": selected_model or "N/A",
            "Accuracy": _format_percent(raw_metrics.get("accuracy")),
            "Macro F1": _format_number(raw_metrics.get("macro_f1")),
            "Log-loss": _format_number(raw_metrics.get("log_loss")),
            "Brier": _format_number(raw_metrics.get("brier")),
            "ECE": _format_number(raw_metrics.get("ece")),
            "MAE goles A": _format_number(raw_metrics.get("mae_goals_a")),
            "MAE goles B": _format_number(raw_metrics.get("mae_goals_b")),
        },
        {
            "Corte": "Test calibrado",
            "Modelo": selected_model or "N/A",
            "Accuracy": _format_percent(calibrated_metrics.get("accuracy")),
            "Macro F1": _format_number(calibrated_metrics.get("macro_f1")),
            "Log-loss": _format_number(calibrated_metrics.get("log_loss")),
            "Brier": _format_number(calibrated_metrics.get("brier")),
            "ECE": _format_number(calibrated_metrics.get("ece")),
            "MAE goles A": _format_number(calibrated_metrics.get("mae_goals_a")),
            "MAE goles B": _format_number(calibrated_metrics.get("mae_goals_b")),
        },
        {
            "Corte": "Test servido",
            "Modelo": "DL calibrado + Dixon-Coles",
            "Accuracy": _format_percent(served_metrics.get("accuracy")),
            "Macro F1": _format_number(served_metrics.get("macro_f1")),
            "Log-loss": _format_number(served_metrics.get("log_loss")),
            "Brier": _format_number(served_metrics.get("brier")),
            "ECE": _format_number(served_metrics.get("ece")),
            "MAE goles A": _format_number(served_metrics.get("mae_goals_a")),
            "MAE goles B": _format_number(served_metrics.get("mae_goals_b")),
        },
    ]
    st.dataframe(pd.DataFrame(metric_rows), hide_index=True, width="stretch")

    comparison = pd.DataFrame(_model_comparison_rows(metrics, selected_model))
    if not comparison.empty:
        st.subheader("Seleccion entre candidatos")
        display_comparison = comparison.copy()
        for column in ("Log-loss CV", "Brier CV", "Macro F1 CV"):
            display_comparison[column] = display_comparison[column].map(lambda value: _format_number(value))
        st.dataframe(display_comparison, hide_index=True, width="stretch")
        figure = px.bar(
            comparison.sort_values("Log-loss CV", ascending=False),
            x="Log-loss CV",
            y="Modelo",
            orientation="h",
            color="Seleccionado",
            color_discrete_map={"Si": "#2ec4a6", "": "#6b7280"},
            labels={"Log-loss CV": "Log-loss promedio CV"},
        )
        figure.update_layout(showlegend=False)
        st.plotly_chart(style_plotly_figure(figure), width="stretch")

    folds = pd.DataFrame(_fold_rows(model_metrics))
    if not folds.empty:
        st.subheader("Estabilidad por fold temporal")
        display_folds = folds.copy()
        display_folds["Accuracy"] = display_folds["Accuracy"].map(_format_percent)
        for column in ("Macro F1", "Log-loss", "Brier", "ECE"):
            display_folds[column] = display_folds[column].map(_format_number)
        st.dataframe(display_folds, hide_index=True, width="stretch")

    if "confusion_as_percent" not in st.session_state:
        st.session_state["confusion_as_percent"] = False
    confusion = _confusion_frame(served_metrics, percent=st.session_state["confusion_as_percent"])
    if not confusion.empty:
        st.subheader("Matriz de confusion del test servido")
        if st.button(
            "Ver conteos" if st.session_state["confusion_as_percent"] else "Ver porcentajes",
            key="toggle_confusion_percent",
        ):
            st.session_state["confusion_as_percent"] = not st.session_state["confusion_as_percent"]
            st.rerun()
        st.caption(
            "Filas: resultado real. Columnas: prediccion del modelo. "
            "En porcentajes, cada fila suma 100% para ver como se distribuyen los errores por clase real."
        )
        figure = confusion_heatmap(served_metrics, percent=st.session_state["confusion_as_percent"])
        if figure:
            st.plotly_chart(figure, width="stretch")
        with st.expander("Ver matriz en tabla"):
            st.dataframe(confusion, hide_index=True, width="stretch")
        if _has_no_predicted_draws(served_metrics):
            st.warning("Limitacion observada: en este test el modelo servido no predijo empates.")

    st.subheader("Diagnostico bayesiano y Dixon-Coles")
    st.caption(
        "NUTS audit valida la estabilidad de una muestra bayesiana de auditoria. "
        "Dixon-Coles produccion es el ajuste que transforma probabilidades de resultado en distribuciones de goles."
    )
    with st.expander("Como leer estas metricas"):
        st.markdown(
            """
            **ELBO final**: objetivo de optimizacion variacional. Sirve para monitorear el ajuste; no se compara
            directamente contra accuracy o log-loss.

            **Iteraciones**: pasos usados para ajustar el componente. En Dixon-Coles, 50000 indica una corrida larga
            del ajuste variacional.

            **Cambio ELBO**: cambio relativo al final del entrenamiento. Valores cercanos a 0 sugieren convergencia;
            aqui 0.0005 indica que el ajuste ya casi no estaba cambiando.

            **Max R-hat**: diagnostico de convergencia de NUTS. Debe estar cerca de 1.00; 1.000 es una buena senal.

            **Min ESS**: muestras efectivas minimas de NUTS. Mas alto es mejor; 1408 da una auditoria estable.

            **N/A**: no aplica para ese metodo. NUTS reporta R-hat/ESS, mientras Dixon-Coles variacional reporta
            ELBO/iteraciones.
            """
        )
    diagnostic_rows = [
        {
            "Componente": "NUTS audit",
            "Iteraciones": "N/A",
            "ELBO final": "N/A",
            "Cambio ELBO": "N/A",
            "Max R-hat": _format_number(nuts.get("max_rhat")),
            "Min ESS": _format_number(nuts.get("min_ess_bulk"), 0),
        },
        {
            "Componente": "Dixon-Coles produccion",
            "Iteraciones": _format_number(dixon_production.get("iterations"), 0),
            "ELBO final": _format_number(dixon_production.get("final_elbo")),
            "Cambio ELBO": _format_number(dixon_production.get("elbo_relative_change"), 4),
            "Max R-hat": "N/A",
            "Min ESS": "N/A",
        },
    ]
    st.dataframe(pd.DataFrame(diagnostic_rows), hide_index=True, width="stretch")

    if state:
        status_counts = Counter(match.status for match in state.matches.values())
        pending = sorted(
            [match for match in state.matches.values() if not match.is_finished],
            key=lambda match: _parse_kickoff(match.kickoff),
        )
        active_phase = pending[0].phase if pending else ""
        st.subheader("Contexto de aplicacion actual")
        st.caption(
            f"El modelo se esta aplicando con {status_counts.get('finished', 0)}/104 partidos oficiales "
            f"ya fijados; fase actual: {_phase_label(active_phase)}."
        )
        if update_status and not update_status.get("ok", True):
            st.warning(f"Ultima consulta FIFA fallida: {update_status.get('attempted_at')}")


DATA_PIPELINE_DOT = r"""
digraph flujo {
  rankdir=TB;
  bgcolor="transparent";
  node [shape=box, style="rounded,filled", fontname="Helvetica",
        fontsize=10, color="#2ec4a6", fillcolor="#2ec4a6",
        fontcolor="#0b141a", margin="0.18,0.10"];
  edge [color="#8894a0", penwidth=1.4, arrowsize=0.7];

  raw    [label="data/raw/*\nKaggle · FIFA API v3 · Polymarket"];
  proc   [label="data/processed\nmatches.parquet + sequences.npz"];
  model  [label="artifacts/selected_model.keras\nbackbone DL (MLP · LSTM · GRU)"];
  stat   [label="posteriores bayesianos\ncalibración + Dixon–Coles"];
  bundle [label="inference_bundle.joblib\nartifact_manifest v2"];
  app    [label="Streamlit · simulador\nMonte Carlo", fillcolor="#1f6f5c", fontcolor="white"];

  raw -> proc -> model -> stat -> bundle -> app;
}
"""


def render_data_section() -> None:
    st.header("Datos usados")
    st.caption("De donde viene cada fuente y que se hizo con ella.")

    data_rows = [
        {
            "Fuente": "Partidos internacionales",
            "Procedencia": "Kaggle · martj42/international-football-results-from-1872-to-2017",
            "Cobertura": "49,505 partidos · 1872-2026",
            "Que se hizo": "Base historica para resultados, goles, forma reciente, localia neutral y enfrentamientos directos.",
        },
        {
            "Fuente": "Mundiales historicos",
            "Procedencia": "Kaggle · piterfm/fifa-football-world-cup",
            "Cobertura": "964 partidos · Mundiales 1930-2022 + calendario/ranking 2026",
            "Que se hizo": "Contexto de campeones, partidos mundialistas y calendario base 2026.",
        },
        {
            "Fuente": "Cartas / ratings FC 24",
            "Procedencia": "Kaggle · stefanoleone992/ea-sports-fc-24-complete-player-dataset",
            "Cobertura": "Jugadores y equipos FC 24",
            "Que se hizo": "Se agregaron atributos por seleccion: overall, pace, shooting, defending, physical e imputacion si faltaba plantel.",
        },
        {
            "Fuente": "Ranking FIFA",
            "Procedencia": "Kaggle · cashncarry/fifaworldranking",
            "Cobertura": "Snapshots ranking FIFA hasta 2024-06",
            "Que se hizo": "Se uso como senal previa de fuerza relativa: ranking A, ranking B y diferencia de ranking.",
        },
        {
            "Fuente": "Estado oficial Mundial 2026",
            "Procedencia": "FIFA API v3 · calendar/matches",
            "Cobertura": "104 partidos del torneo 2026",
            "Que se hizo": "Se normalizaron resultados, horarios, fase, dependencias y ganadores para bloquear partidos reales.",
        },
        {
            "Fuente": "Mercado Polymarket",
            "Procedencia": "Gamma API + CLOB Polymarket",
            "Cobertura": "Mercados 1-X-2 a 90 minutos para cruces pendientes",
            "Que se hizo": "Se normalizaron precios Yes, liquidez y spread; se usan como opinion de mercado cuando pasan filtros.",
        },
    ]
    st.dataframe(
        pd.DataFrame(data_rows),
        hide_index=True,
        width="stretch",
        height=560,
        row_height=84,
        column_config={
            "Fuente": st.column_config.TextColumn(width="small"),
            "Procedencia": st.column_config.TextColumn(width="medium"),
            "Cobertura": st.column_config.TextColumn(width="small"),
            "Que se hizo": st.column_config.TextColumn(width="large"),
        },
    )

    left, right = st.columns([1, 1])
    with left:
        st.subheader("Flujo de datos")
        st.graphviz_chart(DATA_PIPELINE_DOT, width="stretch")
        st.caption("Del dato crudo al simulador: cada etapa versiona su artefacto.")
    with right:
        st.subheader("Transformacion")
        st.dataframe(
            pd.DataFrame(
                [
                    {"Paso": "Descarga", "Resultado": "Datos crudos versionados fuera de Git."},
                    {"Paso": "Construccion", "Resultado": "Dataset tabular y secuencias temporales."},
                    {"Paso": "Entrenamiento", "Resultado": "Backbone DL, calibracion y Dixon-Coles."},
                    {"Paso": "Validacion", "Resultado": "CV temporal, test bloqueado y diagnosticos."},
                    {"Paso": "Servicio", "Resultado": "Dashboard con simulacion Monte Carlo."},
                ]
            ),
            hide_index=True,
            width="stretch",
        )


def _write_update_status(ok: bool, error: str | None = None, **extra: object) -> None:
    status_path = STATE_PATH.with_name("tournament_state_update.json")
    payload = {"attempted_at": utc_now(), "ok": ok, "error": error, **extra}
    status_path.parent.mkdir(parents=True, exist_ok=True)
    status_path.write_text(json.dumps(payload, indent=2, ensure_ascii=False), encoding="utf-8")


def _refresh_polymarket_snapshot(state) -> str | None:
    command = [
        sys.executable,
        str(Path(__file__).with_name("scripts") / "generate_polymarket_snapshot.py"),
        "--output", str(ARTIFACTS_DIR / "polymarket_snapshot.json"),
        "--tournament-state", str(STATE_PATH),
    ]
    env = os.environ.copy()
    src_path = str(Path(__file__).resolve().parent / "src")
    env["PYTHONPATH"] = src_path + (os.pathsep + env["PYTHONPATH"] if env.get("PYTHONPATH") else "")
    result = subprocess.run(command, check=False, capture_output=True, text=True, env=env)
    if not result.returncode:
        return None
    refresh_error = (result.stderr or result.stdout).strip()
    snapshot_path = ARTIFACTS_DIR / "polymarket_snapshot.json"
    if snapshot_path.exists():
        retained = filter_pending_markets(load_snapshot(snapshot_path), state)
        old = json.loads(snapshot_path.read_text(encoding="utf-8"))
        old["markets"] = [dataclasses.asdict(market) for market in retained]
        old["tournament_state_hash"] = state.to_dict()["hash"]
        old["refresh_error"] = refresh_error
        snapshot_path.write_text(json.dumps(old, indent=2, ensure_ascii=False), encoding="utf-8")
    return refresh_error


def refresh_official_state():
    try:
        state = fetch_fifa_state(FIFA_SOURCE_URL)
        approved = load_tournament_state(STATE_PATH)
        validate_state(state, approved)
        write_state(state, CANDIDATE_PATH)
        promoted = approve_candidate(CANDIDATE_PATH, STATE_PATH)
        polymarket_error = _refresh_polymarket_snapshot(promoted)
        _write_update_status(True, None, polymarket_error=polymarket_error)
        return promoted, polymarket_error
    except Exception as error:
        CANDIDATE_PATH.unlink(missing_ok=True)
        _write_update_status(False, str(error))
        raise

# The approved state mtime invalidates both the market snapshot adapter and
# the simulator cache after a promotion, without retraining the backbone.
state_version = STATE_PATH.stat().st_mtime_ns if STATE_PATH.exists() else 0
predictor, simulator, model_mode = resources(state_version)
default_groups = load_groups()
all_default_teams = [team for group in GROUPS for team in default_groups[group]]
try:
    tournament_state = load_tournament_state(STATE_PATH)
except (OSError, ValueError, json.JSONDecodeError) as error:
    tournament_state = None
    st.error(f"El estado oficial no se pudo cargar; se usa simulacion completa: {error}")
status_path = STATE_PATH.with_name("tournament_state_update.json")
update_status = None
if status_path.exists():
    update_status = json.loads(status_path.read_text(encoding="utf-8"))
    if not update_status.get("ok", True):
        st.warning(
            f"La ultima consulta FIFA fallo ({update_status.get('attempted_at')}). "
            "Se sirve el ultimo snapshot aprobado."
        )

render_app_header(model_mode, tournament_state, update_status)

predictor_tab, groups_tab, bracket_tab, validation_tab, champions_tab, data_tab = st.tabs(
    ["🎯 Predictor", "🧩 Grupos", "🏆 Eliminacion", "📊 Confianza modelo", "⭐ Campeones", "🗂️ Datos"]
)

with predictor_tab:
    st.header("¿Que puede pasar en este partido?")
    first, second = st.columns(2)
    team_a = first.selectbox("Seleccion A", all_default_teams, index=0, format_func=team_label)
    team_b = second.selectbox("Seleccion B", all_default_teams, index=1, format_func=team_label)
    if team_a == team_b:
        st.error("Elija dos selecciones diferentes.")
    else:
        prediction = predictor.predict_match(team_a, team_b)
        render_probability_cards(prediction)
        render_market_opinion(prediction)

with st.sidebar:
    official_count = len(tournament_state.finished_matches) if tournament_state else 0
    pending_count = (104 - official_count) if tournament_state else 104
    updated = (tournament_state.approved_at or tournament_state.source_updated_at or tournament_state.generated_at) if tournament_state else "N/A"
    update_label = "OK" if not update_status or update_status.get("ok", True) else "fallback"
    next_title, next_subtitle = next_match_label(tournament_state)
    model_name, model_ece, model_accuracy = sidebar_model_summary()
    market_count = sidebar_market_count()
    st.markdown(
        f"""
        <div class="sidebar-hero">
            <div class="sidebar-kicker">Match control</div>
            <div class="sidebar-next">{next_title}</div>
            <div class="sidebar-line">{next_subtitle}</div>
            <div class="sidebar-pill-row">
                <span class="sidebar-pill">Modelo {model_name}</span>
                <span class="sidebar-pill">ECE {model_ece}</span>
                <span class="sidebar-pill">Acc {model_accuracy}</span>
                <span class="sidebar-pill">Mercados {market_count}</span>
            </div>
        </div>
        """,
        unsafe_allow_html=True,
    )
    st.markdown(
        f"""
        <div class="sidebar-card">
            <div class="sidebar-title">Datos oficiales</div>
            <div class="sidebar-line">Fuente: FIFA API v3</div>
            <div class="sidebar-line">Snapshot: {official_count}/104 jugados · {pending_count} pendientes</div>
            <div class="sidebar-line">Actualizado: {updated}</div>
            <div class="sidebar-line">Estado: {update_label}</div>
        </div>
        """,
        unsafe_allow_html=True,
    )
    if st.button("Actualizar datos", type="secondary", width="stretch"):
        with st.spinner("Consultando FIFA y recalculando el torneo..."):
            try:
                promoted_state, market_error = refresh_official_state()
                st.session_state["overrides"] = {}
                resources.clear()
                message = f"Actualizado: {len(promoted_state.finished_matches)} partidos oficiales."
                if market_error:
                    st.warning(message + " Polymarket no se pudo refrescar; se filtró el snapshot anterior.")
                else:
                    st.success(message)
                st.rerun()
            except Exception as error:
                st.error(f"No se modifico el estado aprobado: {error}")
    st.markdown(
        """
        <div class="sidebar-card">
            <div class="sidebar-title">Simulacion</div>
            <div class="sidebar-line">Ajuste el numero de corridas y semilla para controlar estabilidad.</div>
        </div>
        """,
        unsafe_allow_html=True,
    )
    # El tier gratis de Streamlit Cloud tiene CPU/RAM limitados; 10k/20k con
    # incertidumbre posterior puede colgar o agotar memoria en una demo. Se
    # capa a 2.000 por defecto y se exponen las opciones pesadas solo cuando
    # MUNDIAL_UNLOCK_HEAVY_RUNS=1 (local con recursos suficientes).
    run_options = [100, 250, 500, 1_000, 2_000]
    if os.environ.get("MUNDIAL_UNLOCK_HEAVY_RUNS") == "1":
        run_options += [10_000, 20_000]
    runs = st.select_slider("Simulaciones", options=run_options, value=2_000)
    seed = st.number_input("Semilla", min_value=1, value=2026, step=1)

with bracket_tab:
    st.header("Camino a la final")
    simulate_from_zero = st.toggle(
        "Simular desde cero",
        value=False,
        help="Ignora resultados oficiales aprobados y desbloquea grupos/cruces para simular el torneo completo.",
    )

active_tournament_state = None if simulate_from_zero else tournament_state

group_editor_enabled = active_tournament_state is None
edited_groups = groups_from_session(default_groups, enabled=group_editor_enabled)

try:
    validate_groups(edited_groups)
    groups_valid = True
except ValueError as error:
    groups_valid = False
    with groups_tab:
        st.error(str(error))

overrides = st.session_state.get("overrides", {})
simulation = run_tournament_simulation(
    simulator,
    edited_groups,
    dict(overrides),
    int(runs),
    int(seed),
    active_tournament_state is not None,
    state_version,
    active_tournament_state,
) if groups_valid else None

with groups_tab:
    st.header("Fase de grupos")
    if simulation:
        st.subheader("Proyeccion de los grupos")
        for row_start in range(0, 12, 3):
            columns = st.columns(3)
            for offset, group in enumerate(GROUPS[row_start:row_start + 3]):
                table = pd.DataFrame([
                    {
                        "Seleccion": team_label(item.team),
                        "Puntos": round(item.expected_points, 2),
                        "GF": round(item.expected_goals_for, 2),
                        "GC": round(item.expected_goals_against, 2),
                        "Clasifica": f"{item.qualification_probability:.1%}",
                    }
                    for item in simulation.group_tables[group]
                ])
                columns[offset].markdown(f"#### Grupo {group}")
                columns[offset].dataframe(table, hide_index=True, width="stretch")
    with st.expander("Reorganizar selecciones", expanded=False):
        st.info("Una seleccion solo puede aparecer una vez entre los doce grupos.")
        if tournament_state and not simulate_from_zero:
            st.caption("Los grupos quedan bloqueados mientras se aplica el estado oficial aprobado.")
        elif tournament_state and simulate_from_zero:
            st.caption("Modo desde cero activo: se ignoran resultados oficiales y los grupos quedan editables.")
        groups_from_editor(default_groups, disabled=not group_editor_enabled)

with bracket_tab:
    if simulate_from_zero:
        st.info("Modo desde cero activo: el cuadro se simula sin resultados oficiales bloqueados.")
    if not simulation:
        st.warning("Corrija la configuracion de grupos para generar el cuadro.")
    else:
        for round_name in ("Ronda de 32", "Octavos", "Cuartos", "Semifinal", "Tercer puesto", "Final"):
            st.subheader(round_name)
            round_matches = [match for match in simulation.bracket if match.round_name == round_name]
            columns = st.columns(2)
            for index, match in enumerate(round_matches):
                forced = " · forzado" if match.forced else ""
                state_label = "real" if match.official else "simulado"
                favorite = match.team_a if match.probability_a >= match.probability_b else match.team_b
                render_match_card(columns[index % 2], match, favorite, state_label, forced)
        with st.expander("Forzar resultados del cuadro mostrado"):
            pending: dict[str, str] = {}
            for match in simulation.bracket:
                if match.official:
                    st.caption(f"{match.match_id}: resultado oficial bloqueado — {team_label(match.winner)}")
                    continue
                options = ["Automatico", match.team_a, match.team_b]
                current = overrides.get(match.match_id, "Automatico")
                if current not in options:
                    current = "Automatico"
                choice = st.selectbox(
                    f"{match.match_id}: {team_label(match.team_a)} vs {team_label(match.team_b)}",
                    options,
                    index=options.index(current),
                    key=f"override_{match.match_id}_{match.team_a}_{match.team_b}",
                    format_func=lambda option: option if option == "Automatico" else team_label(option),
                )
                if choice != "Automatico":
                    pending[match.match_id] = choice
            if st.button("Aplicar resultados y recalcular", type="primary"):
                st.session_state["overrides"] = pending
                st.rerun()
            if st.button("Restablecer resultados"):
                st.session_state["overrides"] = {}
                st.rerun()

with validation_tab:
    render_validation_section(tournament_state, update_status)

with champions_tab:
    st.header("¿Quien levantaria la copa?")
    if not simulation:
        st.warning("Corrija la configuracion de grupos para calcular campeones.")
    else:
        market_used = simulation.metadata.get("market_crossings_used", 0)
        fallback = simulation.metadata.get("fallback_count", 0)
        if market_used + fallback > 0:
            st.caption(f"Polymarket: {market_used} cruces con mercado, {fallback} sin mercado (DL puro)")
        top = list(simulation.champion_probabilities.items())[:10]
        eliminated = set(simulation.metadata.get("eliminated_teams", []))
        if eliminated:
            st.subheader("Selecciones activas y eliminadas")
            active = [team for team, probability in simulation.champion_probabilities.items() if team not in eliminated]
            metric_active, metric_out = st.columns(2)
            metric_active.metric("Activas", len(active))
            metric_out.metric("Eliminadas", len(eliminated))

            with st.expander(f"Ver probabilidades de las {len(active)} activas"):
                active_frame = pd.DataFrame(
                    [
                        {
                            "Seleccion": team_label(team),
                            "Probabilidad": f"{simulation.champion_probabilities[team]:.2%}",
                        }
                        for team in active
                    ]
                )
                st.dataframe(active_frame, hide_index=True, width="stretch")

            with st.expander(f"Ver eliminadas ({len(eliminated)})"):
                eliminated_frame = pd.DataFrame(
                    [
                        {"Seleccion": team_label(team), "Probabilidad": "0.00%"}
                        for team in sorted(eliminated)
                    ]
                )
                st.dataframe(eliminated_frame, hide_index=True, width="stretch")
        intervals = simulation.metadata.get("champion_confidence_intervals_95", {})
        chart = pd.DataFrame([
            {
                "Seleccion": team_label(team), "Probabilidad": probability,
                "IC inferior": intervals.get(team, (probability, probability))[0],
                "IC superior": intervals.get(team, (probability, probability))[1],
            }
            for team, probability in top
        ]).sort_values("Probabilidad")
        # El IC 95% (error Monte Carlo) va como barras de error sobre cada
        # barra; asi no hace falta repetir una tabla de probabilidades aparte.
        chart["err_mas"] = (chart["IC superior"] - chart["Probabilidad"]).clip(lower=0)
        chart["err_menos"] = (chart["Probabilidad"] - chart["IC inferior"]).clip(lower=0)
        figure = px.bar(
            chart, x="Probabilidad", y="Seleccion", orientation="h",
            text=chart["Probabilidad"].map(lambda value: f"{value:.1%}"),
            error_x="err_mas", error_x_minus="err_menos",
            custom_data=["IC inferior", "IC superior"],
            labels={"Probabilidad": "Probabilidad de ser campeon"},
            color_discrete_sequence=["#2ec4a6"],
        )
        figure.update_xaxes(tickformat=".0%")
        figure.update_traces(
            error_x_color="#9aa7b2",
            hovertemplate=(
                "%{y}<br>Probabilidad: %{x:.2%}"
                "<br>IC 95%: %{customdata[0]:.2%} – %{customdata[1]:.2%}<extra></extra>"
            ),
        )
        figure.update_layout(showlegend=False)
        st.caption("Barras de error = intervalo Monte Carlo al 95% (Wilson).")
        st.plotly_chart(style_plotly_figure(figure), width="stretch")
        titles, source = load_world_cup_titles()
        context = pd.DataFrame(
            [{"Seleccion": team_label(team), "Titulos mundiales": titles.get(team, 0)} for team, _ in top]
        )
        st.subheader("Contexto historico")
        st.dataframe(context, hide_index=True, width="stretch")
        st.caption(source)

with data_tab:
    render_data_section()
