"""Dashboard en espanol del sistema Mundial 2026."""

from __future__ import annotations

import json

import pandas as pd
import plotly.express as px
import streamlit as st

from mundial.config import load_groups
from mundial.history import load_world_cup_titles
from mundial.inference import load_predictor
from mundial.simulation import GROUPS, TournamentSimulator, validate_groups
from mundial.tournament_state import STATE_PATH, load_tournament_state

st.set_page_config(page_title="Inteligencia Mundial 2026", page_icon="⚽", layout="wide")


@st.cache_resource
def resources(state_version: int):
    predictor, mode = load_predictor()
    return predictor, TournamentSimulator(predictor), mode


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
                )
                for position in range(4)
            ]
    return edited


def render_probability_cards(prediction) -> None:
    columns = st.columns(3)
    columns[0].metric(f"Gana {prediction.team_a}", f"{prediction.prob_a:.1%}")
    columns[1].metric("Empate", f"{prediction.prob_draw:.1%}")
    columns[2].metric(f"Gana {prediction.team_b}", f"{prediction.prob_b:.1%}")
    st.subheader(f"Marcador mas probable: {prediction.likely_score[0]} – {prediction.likely_score[1]}")
    bp = getattr(prediction, "base_probabilities", None)
    mp = getattr(prediction, "market_probabilities", None)
    mw = getattr(prediction, "market_weight", None)
    if bp is not None:
        if mp is not None and mw is not None and mw > 0:
            with st.expander("Detalle: modelo vs mercado"):
                col1, col2, col3 = st.columns(3)
                col1.caption("Modelo DL")
                col1.write(f"{prediction.team_a}: {bp[0]:.1%} | Empate: {bp[1]:.1%} | {prediction.team_b}: {bp[2]:.1%}")
                col2.caption("Polymarket")
                col2.write(f"{prediction.team_a}: {mp[0]:.1%} | Empate: {mp[1]:.1%} | {prediction.team_b}: {mp[2]:.1%}")
                col3.caption("Combinado")
                alpha_str = f"α={mw:.2f}" if mw is not None else "N/A"
                col3.write(f"Peso {alpha_str} | Capturado: {getattr(prediction, 'market_as_of', None) or 'N/A'}")
        else:
            st.info("Este cruce no usa una cotización Polymarket válida; se sirve el modelo DL.")


st.title("⚽ Inteligencia deportiva — Mundial 2026")
# The approved state mtime invalidates both the market snapshot adapter and
# the simulator cache after a promotion, without retraining the backbone.
state_version = STATE_PATH.stat().st_mtime_ns if STATE_PATH.exists() else 0
predictor, simulator, model_mode = resources(state_version)
st.caption(model_mode)
default_groups = load_groups()
all_default_teams = [team for group in GROUPS for team in default_groups[group]]
try:
    tournament_state = load_tournament_state(STATE_PATH)
except (OSError, ValueError, json.JSONDecodeError) as error:
    tournament_state = None
    st.error(f"El estado oficial no se pudo cargar; se usa simulacion completa: {error}")
if tournament_state:
    updated = tournament_state.approved_at or tournament_state.source_updated_at or tournament_state.generated_at
    st.caption(f"Estado oficial actualizado hasta {updated} · [Fuente FIFA]({tournament_state.source_url})")
status_path = STATE_PATH.with_name("tournament_state_update.json")
if status_path.exists():
    update_status = json.loads(status_path.read_text(encoding="utf-8"))
    if not update_status.get("ok", True):
        st.warning(
            f"La ultima consulta FIFA fallo ({update_status.get('attempted_at')}). "
            "Se sirve el ultimo snapshot aprobado."
        )

predictor_tab, groups_tab, bracket_tab, champions_tab = st.tabs(
    ["Predictor de partidos", "Fase de grupos", "Eliminacion directa", "Campeones probables"]
)

with predictor_tab:
    st.header("¿Que puede pasar en este partido?")
    first, second = st.columns(2)
    team_a = first.selectbox("Seleccion A", all_default_teams, index=0)
    team_b = second.selectbox("Seleccion B", all_default_teams, index=1)
    if team_a == team_b:
        st.error("Elija dos selecciones diferentes.")
    else:
        render_probability_cards(predictor.predict_match(team_a, team_b))

with st.sidebar:
    st.header("Configuracion")
    runs = st.select_slider(
        "Simulaciones", options=[100, 250, 500, 1_000, 2_000, 10_000, 20_000], value=2_000
    )
    seed = st.number_input("Semilla", min_value=1, value=2026, step=1)
    st.caption("Mas simulaciones producen porcentajes mas estables.")

with groups_tab:
    st.header("Reorganice las selecciones")
    st.info("Una seleccion solo puede aparecer una vez entre los doce grupos.")
    if tournament_state:
        st.caption("Los grupos quedan bloqueados mientras se aplica el estado oficial aprobado.")
    edited_groups = groups_from_editor(default_groups, disabled=tournament_state is not None)

try:
    validate_groups(edited_groups)
    groups_valid = True
except ValueError as error:
    groups_valid = False
    with groups_tab:
        st.error(str(error))

overrides = st.session_state.get("overrides", {})
simulation = simulator.simulate(
    edited_groups, overrides=overrides, runs=int(runs), seed=int(seed), tournament_state=tournament_state
) if groups_valid else None

with groups_tab:
    if simulation:
        st.subheader("Proyeccion de los grupos")
        for row_start in range(0, 12, 3):
            columns = st.columns(3)
            for offset, group in enumerate(GROUPS[row_start:row_start + 3]):
                table = pd.DataFrame([
                    {
                        "Seleccion": item.team,
                        "Puntos": round(item.expected_points, 2),
                        "GF": round(item.expected_goals_for, 2),
                        "GC": round(item.expected_goals_against, 2),
                        "Clasifica": f"{item.qualification_probability:.1%}",
                    }
                    for item in simulation.group_tables[group]
                ])
                columns[offset].markdown(f"#### Grupo {group}")
                columns[offset].dataframe(table, hide_index=True, width="stretch")

with bracket_tab:
    st.header("Camino a la final")
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
                columns[index % 2].markdown(
                    f"**{match.match_id}** — {match.team_a} ({match.probability_a:.0%}) "
                    f"vs {match.team_b} ({match.probability_b:.0%})  \n"
                    f"Ganador: **{match.winner}** · {state_label}{forced}"
                )
        with st.expander("Forzar resultados del cuadro mostrado"):
            pending: dict[str, str] = {}
            for match in simulation.bracket:
                if match.official:
                    st.caption(f"{match.match_id}: resultado oficial bloqueado — {match.winner}")
                    continue
                options = ["Automatico", match.team_a, match.team_b]
                current = overrides.get(match.match_id, "Automatico")
                if current not in options:
                    current = "Automatico"
                choice = st.selectbox(
                    f"{match.match_id}: {match.team_a} vs {match.team_b}",
                    options,
                    index=options.index(current),
                    key=f"override_{match.match_id}_{match.team_a}_{match.team_b}",
                )
                if choice != "Automatico":
                    pending[match.match_id] = choice
            if st.button("Aplicar resultados y recalcular", type="primary"):
                st.session_state["overrides"] = pending
                st.rerun()
            if st.button("Restablecer resultados"):
                st.session_state["overrides"] = {}
                st.rerun()

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
            col_active, col_out = st.columns(2)
            col_active.caption(f"Activas ({len(active)})")
            col_active.write(", ".join(active))
            col_out.caption(f"Eliminadas ({len(eliminated)}) · probabilidad de campeon 0%")
            col_out.write(", ".join(sorted(eliminated)))
        intervals = simulation.metadata.get("champion_confidence_intervals_95", {})
        chart = pd.DataFrame([
            {
                "Seleccion": team, "Probabilidad": probability,
                "IC inferior": intervals.get(team, (probability, probability))[0],
                "IC superior": intervals.get(team, (probability, probability))[1],
            }
            for team, probability in top
        ])
        figure = px.bar(
            chart.sort_values("Probabilidad"), x="Probabilidad", y="Seleccion", orientation="h",
            text=chart.sort_values("Probabilidad")["Probabilidad"].map(lambda value: f"{value:.1%}"),
            labels={"Probabilidad": "Probabilidad de ser campeon"},
        )
        figure.update_xaxes(tickformat=".0%")
        figure.update_layout(showlegend=False)
        st.plotly_chart(figure, width="stretch")
        st.dataframe(
            chart.assign(
                Probabilidad=chart["Probabilidad"].map(lambda value: f"{value:.2%}"),
                **{
                    "IC 95%": chart.apply(
                        lambda row: f"{row['IC inferior']:.2%} – {row['IC superior']:.2%}", axis=1
                    )
                },
            )[["Seleccion", "Probabilidad", "IC 95%"]],
            hide_index=True, width="stretch",
        )
        titles, source = load_world_cup_titles()
        context = pd.DataFrame(
            [{"Seleccion": team, "Titulos mundiales": titles.get(team, 0)} for team, _ in top]
        )
        st.subheader("Contexto historico")
        st.dataframe(context, hide_index=True, width="stretch")
        st.caption(source)
