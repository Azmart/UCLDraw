# draw.py
# Streamlit simulator for the UEFA Champions League (league phase) draw
# Now supports importing pots & country mappings from a local data.py module,
# in addition to JSON input. JSON has priority when selected.
#
# See instructions within the UI sidebar.

from __future__ import annotations

import json
import random
from collections import defaultdict, Counter
from typing import Dict, List, Tuple, Set, Optional

import streamlit as st

# Try importing local data.py (optional)
DATA_MODULE = None
try:
    import data as DATA_MODULE  # must sit next to draw.py
except Exception:
    DATA_MODULE = None

# ----------------------------
# Utilities: simple validation
# ----------------------------

def normalize_input(data: dict) -> Tuple[Dict[int, List[str]], Dict[str, str]]:
    """
    Accepts dict that contains:
      - "pots": {"1":[...],"2":[...],"3":[...],"4":[...]}  (or int keys)
      - AND EITHER "team_to_country" OR "participants_by_country"
    Returns:
      pots: {1:[teams],2:[teams],3:[teams],4:[teams]}
      team_to_country: {"Team":"Country"}
    Raises ValueError if schema inconsistent.
    """
    if "pots" not in data or not isinstance(data["pots"], dict):
        raise ValueError("Input must contain a 'pots' dictionary with keys '1','2','3','4'.")

    raw_pots = data["pots"]
    # Normalize pot keys to int 1..4
    pots: Dict[int, List[str]] = {}
    for k in (1, 2, 3, 4):
        if str(k) in raw_pots:
            pots[k] = list(raw_pots[str(k)])
        elif k in raw_pots:
            pots[k] = list(raw_pots[k])
        else:
            raise ValueError(f"Pot '{k}' missing from 'pots'.")

    # Validate pot sizes
    for k, teams in pots.items():
        if len(teams) != 9:
            raise ValueError(f"Pot {k} must contain exactly 9 teams (found {len(teams)}).")

    # Validate uniqueness across all pots
    all_teams = sum(pots.values(), [])
    if len(all_teams) != len(set(all_teams)):
        dupes = [t for t, c in Counter(all_teams).items() if c > 1]
        raise ValueError(f"Teams must be unique across all pots. Duplicates: {dupes}")

    # Build team_to_country from either mapping
    if "team_to_country" in data and isinstance(data["team_to_country"], dict):
        team_to_country = dict(data["team_to_country"])
    elif "participants_by_country" in data and isinstance(data["participants_by_country"], dict):
        team_to_country = {}
        for country, teams in data["participants_by_country"].items():
            for t in teams:
                team_to_country[t] = country
    else:
        # Fallback: Unknown for all (still allowed for playing; you can toggle 'same country' rule)
        team_to_country = {t: "Unknown" for t in all_teams}

    # Ensure every team has a mapping
    missing = [t for t in all_teams if t not in team_to_country]
    if missing:
        raise ValueError(f"Missing country mapping for: {missing}")

    return pots, team_to_country


def module_to_input_dict(mod) -> dict:
    """
    Convert a data.py module to the same dict format normalize_input expects.
    Prefers team_to_country if both mappings exist.
    """
    out = {"pots": {}}

    # Pots
    if hasattr(mod, "pots") and isinstance(mod.pots, dict):
        # Convert keys to strings
        for k in (1, 2, 3, 4):
            if str(k) in mod.pots:
                out["pots"][str(k)] = list(mod.pots[str(k)])
            elif k in mod.pots:
                out["pots"][str(k)] = list(mod.pots[k])
            else:
                # Let normalize_input throw a clear error later
                out["pots"][str(k)] = []
    else:
        raise ValueError("data.py must define 'pots' as a dict with 4 pots of 9 teams each.")

    # Countries
    if hasattr(mod, "team_to_country") and isinstance(mod.team_to_country, dict):
        out["team_to_country"] = dict(mod.team_to_country)
    elif hasattr(mod, "participants_by_country") and isinstance(mod.participants_by_country, dict):
        out["participants_by_country"] = dict(mod.participants_by_country)
    else:
        # If neither exists, still return. Unknown allowed.
        pass

    return out

# --------------------------------------
# Core generation: edges + home/away
# --------------------------------------

Edge = Tuple[str, str]  # undirected pair (A,B)
Orientation = Dict[Tuple[str, str], str]  # canonical_pair -> "HOME_TEAM_NAME"

def canonical_pair(a: str, b: str) -> Tuple[str, str]:
    return (a, b) if a < b else (b, a)

def build_full_draw(
    pots: Dict[int, List[str]],
    team_to_country: Dict[str, str],
    forbid_same_country: bool = True,
    seed: Optional[int] = None,
    max_global_tries: int = 200
) -> Tuple[List[Edge], Orientation]: # type: ignore
    """
    Build a complete valid draw satisfying:
      - Each team plays 8 opponents: 2 from each pot (incl. own pot)
      - Optional: no same-country opponents
      - No duplicate pairings
      - Home/Away balanced 4/4 per team
    """
    rng = random.Random(seed)
    teams = sum(pots.values(), [])
    pot_of = {t: p for p, lst in pots.items() for t in lst}

    needs = {t: {1: 2, 2: 2, 3: 2, 4: 2} for t in teams}
    adj: Dict[str, Set[str]] = {t: set() for t in teams}

    # Tasks: 9 edges within each pot; 18 edges for each cross-pot pair
    tasks = []
    for p in (1, 2, 3, 4):
        tasks.append((p, p, 9))
    for p in (1, 2, 3, 4):
        for q in range(p + 1, 5):
            tasks.append((p, q, 18))

    def capacity(p: int, q: int) -> int:
        A = pots[p]
        B = pots[q]
        if p == q:
            c = 0
            for i in range(len(A)):
                for j in range(i + 1, len(A)):
                    if (not forbid_same_country) or (team_to_country[A[i]] != team_to_country[A[j]]):
                        c += 1
            return c
        c = 0
        for a in A:
            for b in B:
                if (not forbid_same_country) or (team_to_country[a] != team_to_country[b]):
                    c += 1
        return c

    tasks.sort(key=lambda x: (capacity(x[0], x[1]), x[0] == x[1]))  # tightest first

    for global_try in range(max_global_tries):
        rng.seed(None if seed is None else (seed + global_try))
        edges: List[Edge] = [] # type: ignore
        for t in teams:
            needs[t] = {1: 2, 2: 2, 3: 2, 4: 2}
            adj[t] = set()

        def compatible(a: str, b: str) -> bool:
            if a == b: return False
            if b in adj[a]: return False
            if needs[a][pot_of[b]] <= 0: return False
            if needs[b][pot_of[a]] <= 0: return False
            if forbid_same_country and team_to_country[a] == team_to_country[b]: return False
            return True

        ok = True
        for (p, q, target) in tasks:
            block_stack: List[Edge] = [] # type: ignore

            def eligible(src_pot: int, need_pot: int) -> List[str]:
                return [t for t in pots[src_pot] if needs[t][need_pot] > 0]

            def cands(t: str, opp_pot: int) -> List[str]:
                lst = [u for u in pots[opp_pot] if compatible(t, u)]
                rng.shuffle(lst)
                return lst

            attempts = 0
            while len(block_stack) < target:
                attempts += 1
                left_pot, right_pot = (p, q) if rng.random() < 0.5 else (q, p)

                left_nodes = eligible(left_pot, right_pot)
                rng.shuffle(left_nodes)
                left_nodes.sort(key=lambda t: sum(1 for _ in cands(t, right_pot)))  # hardest first

                placed = False
                for t in left_nodes:
                    options = cands(t, right_pot)
                    if not options:
                        continue
                    options.sort(key=lambda u: sum(1 for _ in cands(u, left_pot)))
                    for u in options:
                        edges.append((t, u))
                        adj[t].add(u); adj[u].add(t)
                        needs[t][pot_of[u]] -= 1
                        needs[u][pot_of[t]] -= 1
                        block_stack.append((t, u))
                        placed = True
                        break
                    if placed:
                        break

                if not placed:
                    if not block_stack:
                        ok = False
                        break
                    bt_t, bt_u = block_stack.pop()
                    edges.pop()
                    adj[bt_t].remove(bt_u); adj[bt_u].remove(bt_t)
                    needs[bt_t][pot_of[bt_u]] += 1
                    needs[bt_u][pot_of[bt_t]] += 1

                if attempts > 20000:
                    ok = False
                    break

            if not ok:
                break

        if not ok:
            continue

        # Sanity check
        if any(len(adj[t]) != 8 for t in teams):
            continue

        # Assign home/away
        orientation = assign_home_away(edges, teams, pots, max_tries=4000, rng=rng)
        if orientation is None:
            continue
        # Optional strict check:
        if not _verify_per_pot_balance(orientation, edges, pots):
            continue


        return edges, orientation

    raise RuntimeError("Failed to generate a valid draw after many attempts. "
                       "Try loosening constraints or changing the seed.")


def assign_home_away(
    edges: List[Tuple[str, str]],
    teams: List[str],
    pots: Dict[int, List[str]],
    max_tries: int = 4000,
    rng: Optional[random.Random] = None
) -> Optional[Orientation]:
    """
    Orient edges so that for every team:
      - exactly 1 HOME and 1 AWAY vs each pot (hence 4 home + 4 away total)
    This prevents cases like a team having two HOME fixtures vs the same pot.

    Returns:
      orientation: {canonical_pair(teamA, teamB) -> "HOME_TEAM_NAME"} or None if failed.
    """
    R = rng or random.Random()
    pot_of = {t: p for p, lst in pots.items() for t in lst}

    for _ in range(max_tries):
        # For every team and opponent-pot, we need exactly one HOME and one AWAY
        home_needed = {t: {1: 1, 2: 1, 3: 1, 4: 1} for t in teams}
        away_needed = {t: {1: 1, 2: 1, 3: 1, 4: 1} for t in teams}

        orient: Orientation = {}
        order = edges[:]
        R.shuffle(order)

        feasible = True
        for a, b in order:
            pa = pot_of[a]
            pb = pot_of[b]
            # From a's viewpoint the opponent is from pot pb; from b's viewpoint it's pa
            a_home_ok = (home_needed[a][pb] > 0) and (away_needed[b][pa] > 0)
            b_home_ok = (home_needed[b][pa] > 0) and (away_needed[a][pb] > 0)

            if a_home_ok and not b_home_ok:
                # a must be home
                home_needed[a][pb] -= 1
                away_needed[b][pa] -= 1
                orient[canonical_pair(a, b)] = a
            elif b_home_ok and not a_home_ok:
                # b must be home
                home_needed[b][pa] -= 1
                away_needed[a][pb] -= 1
                orient[canonical_pair(a, b)] = b
            elif not a_home_ok and not b_home_ok:
                feasible = False
                break
            else:
                # Both ways are possible — pick the one that relieves tighter needs
                # (simple urgency heuristic; ties broken randomly)
                score_a = (home_needed[a][pb] == 1) + (away_needed[b][pa] == 1)
                score_b = (home_needed[b][pa] == 1) + (away_needed[a][pb] == 1)

                if score_a > score_b or (score_a == score_b and R.random() < 0.5):
                    home_needed[a][pb] -= 1
                    away_needed[b][pa] -= 1
                    orient[canonical_pair(a, b)] = a
                else:
                    home_needed[b][pa] -= 1
                    away_needed[a][pb] -= 1
                    orient[canonical_pair(a, b)] = b

        if not feasible:
            continue

        # All per-pot needs must be met
        if any(v != 0 for t in teams for v in home_needed[t].values()):
            continue
        if any(v != 0 for t in teams for v in away_needed[t].values()):
            continue

        return orient

    return None

def _verify_per_pot_balance(orientation: Orientation, edges: List[Tuple[str, str]], pots: Dict[int, List[str]]) -> bool:
    pot_of = {t: p for p, lst in pots.items() for t in lst}
    home_cnt = {t: {1:0,2:0,3:0,4:0} for p in pots for t in pots[p]}
    away_cnt = {t: {1:0,2:0,3:0,4:0} for p in pots for t in pots[p]}

    for a, b in edges:
        home_team = orientation[canonical_pair(a, b)]
        if home_team == a:
            home_cnt[a][pot_of[b]] += 1
            away_cnt[b][pot_of[a]] += 1
        else:
            home_cnt[b][pot_of[a]] += 1
            away_cnt[a][pot_of[b]] += 1

    return all(home_cnt[t][p] == 1 and away_cnt[t][p] == 1 for p in (1,2,3,4) for t in home_cnt)

# ----------------------------
# Streamlit UI / App state
# ----------------------------

DEFAULT_SAMPLE_JSON = {
    "pots": {
        "1": [f"Sample Team P1-{i:02d}" for i in range(1, 10)],
        "2": [f"Sample Team P2-{i:02d}" for i in range(1, 10)],
        "3": [f"Sample Team P3-{i:02d}" for i in range(1, 10)],
        "4": [f"Sample Team P4-{i:02d}" for i in range(1, 10)],
    },
    "team_to_country": {
        **{f"Sample Team P1-{i:02d}": ["Country A", "Country B", "Country C"][(i-1) % 3] for i in range(1, 10)},
        **{f"Sample Team P2-{i:02d}": ["Country D", "Country E", "Country F"][(i-1) % 3] for i in range(1, 10)},
        **{f"Sample Team P3-{i:02d}": ["Country G", "Country H", "Country I"][(i-1) % 3] for i in range(1, 10)},
        **{f"Sample Team P4-{i:02d}": ["Country J", "Country K", "Country L"][(i-1) % 3] for i in range(1, 10)},
    }
}

def init_session():
    ss = st.session_state
    ss.setdefault("pots_json_text", json.dumps(DEFAULT_SAMPLE_JSON, indent=2))
    ss.setdefault("input_source", "JSON (paste/upload)")  # or "data.py"
    ss.setdefault("pots", None)
    ss.setdefault("team_to_country", None)
    ss.setdefault("forbid_same_country", True)
    ss.setdefault("seed", 42)
    ss.setdefault("edges", None)
    ss.setdefault("orientation", None)
    ss.setdefault("reveal_pointer", 0)
    ss.setdefault("reveal_sequence", [])

def compute_reveal_sequence(pots: Dict[int, List[str]], seed: Optional[int]) -> List[Tuple[int, str]]:
    rng = random.Random(seed)
    seq = []
    for p in (1, 2, 3, 4):
        order = pots[p][:]
        rng.shuffle(order)
        for t in order:
            seq.append((p, t))
    return seq

def start_or_reset():
    ss = st.session_state
    # Decide source based on user's selection (JSON has priority when selected)
    try:
        if ss.input_source.startswith("JSON"):
            data = json.loads(ss.pots_json_text)
        else:
            if DATA_MODULE is None:
                raise ValueError("Could not import data.py. Ensure it exists next to draw.py.")
            data = module_to_input_dict(DATA_MODULE)

        pots, t2c = normalize_input(data)
        ss.pots = pots
        ss.team_to_country = t2c
    except Exception as e:
        st.error(f"Input error: {e}")
        return

    # Build a full valid draw
    try:
        edges, orientation = build_full_draw(
            ss.pots,
            ss.team_to_country,
            forbid_same_country=ss.forbid_same_country,
            seed=ss.seed,
        )
    except Exception as e:
        st.error(str(e))
        return

    ss.edges = edges
    ss.orientation = orientation
    ss.reveal_sequence = compute_reveal_sequence(ss.pots, ss.seed)
    ss.reveal_pointer = 0

def reveal_next():
    ss = st.session_state
    if ss.edges is None:
        st.warning("Please start the draw first.")
        return
    if ss.reveal_pointer < len(ss.reveal_sequence):
        ss.reveal_pointer += 1

def reveal_all():
    ss = st.session_state
    if ss.edges is None:
        st.warning("Please start the draw first.")
        return
    ss.reveal_pointer = len(ss.reveal_sequence)

# ----------------------------
# Rendering helpers
# ----------------------------

Edge = Tuple[str, str]
def edges_for_team(team: str, edges: List[Edge]) -> List[Edge]:        # type: ignore
    return [(a, b) for (a, b) in edges if a == team or b == team]

def is_home(team: str, opponent: str, orientation: Orientation) -> bool:
    key = canonical_pair(team, opponent)
    home_team = orientation.get(key)
    return home_team == team

def format_bowl(bowl_title: str, teams: List[str], revealed: Set[str]):
    remaining = [t for t in teams if t not in revealed]
    st.caption(f"**{bowl_title}** — Remaining balls: {len(remaining)}")
    st.write(",  ".join(remaining) if remaining else "—")

def download_links(pots: Dict[int, List[str]], edges: List[Edge], orientation: Orientation): # type: ignore
    pot_of = {t: p for p, lst in pots.items() for t in lst}
    rows = []
    for t in sum(pots.values(), []):
        for a, b in edges:
            if t not in (a, b): continue
            opp = b if a == t else a
            home = is_home(t, opp, orientation)
            rows.append({
                "Team": t,
                "Pot": pot_of[t],
                "Opponent": opp,
                "Opponent Pot": pot_of[opp],
                "Venue": "Home" if home else "Away"
            })
    json_bytes = json.dumps(rows, indent=2).encode("utf-8")
    st.download_button(
        label="⬇️ Download draw (JSON)",
        data=json_bytes,
        file_name="ucl_draw.json",
        mime="application/json"
    )

def display_data_module_preview():
    """Pretty display of what's inside data.py (if import succeeded)."""
    st.subheader("📦 data.py preview")
    if DATA_MODULE is None:
        st.info("`data.py` not found or failed to import. Place a `data.py` next to `draw.py`.")
        return

    # Build a uniform dict to show
    try:
        data = module_to_input_dict(DATA_MODULE)
        pots, t2c = normalize_input(data)
    except Exception as e:
        st.error(f"data.py format issue: {e}")
        return

    # Headline stats
    all_teams = sum(pots.values(), [])
    colA, colB, colC = st.columns(3)
    colA.metric("Total teams", len(all_teams))
    colB.metric("Pots", "4 (9 each)")
    colC.metric("Distinct countries", len(set(t2c.values())))

    # Tabs
    tabs = st.tabs(["Pots", "Participants by country", "Team → Country"])
    with tabs[0]:
        c1, c2 = st.columns(2)
        with c1:
            st.markdown("**Pot 1**")
            st.table({"Teams": pots[1]})
            st.markdown("**Pot 2**")
            st.table({"Teams": pots[2]})
        with c2:
            st.markdown("**Pot 3**")
            st.table({"Teams": pots[3]})
            st.markdown("**Pot 4**")
            st.table({"Teams": pots[4]})

    with tabs[1]:
        # Build participants_by_country from t2c
        by_country = defaultdict(list)
        for team, ctry in t2c.items():
            by_country[ctry].append(team)
        # Sort for nicer display
        country_rows = []
        for ctry in sorted(by_country.keys()):
            teams = sorted(by_country[ctry])
            country_rows.append({"Country": ctry, "Teams (count)": f"{len(teams)}", "Teams": ", ".join(teams)})
        st.dataframe(country_rows, use_container_width=True)

    with tabs[2]:
        # Simple mapping list
        rows = [{"Team": t, "Country": t2c[t]} for t in sorted(t2c.keys())]
        st.dataframe(rows, use_container_width=True)

# ----------------------------
# App
# ----------------------------

st.set_page_config(page_title="UEFA Champions League Draw Simulator", layout="wide")
init_session()

st.title("🏆 UEFA Champions League — League Phase Draw Simulator")
st.write(
    "Simulate the ceremony step-by-step (Pot 1 → 4) or generate the full draw instantly. "
    "Now supports loading from a **local `data.py`** as well as JSON. "
    "_JSON has priority when selected._"
)

with st.sidebar:
    st.header("Setup")

    # Input source selector (JSON has priority when chosen)
    st.session_state.input_source = st.radio(
        "Choose input source",
        options=["JSON (paste/upload)", "data.py (module)"],
        index=0,
        help="Pick JSON to paste/upload custom data, or use the local data.py file."
    )

    if st.session_state.input_source.startswith("JSON"):
        uploaded = st.file_uploader("Upload JSON (pots + countries)", type=["json"])
        if uploaded:
            st.session_state.pots_json_text = uploaded.read().decode("utf-8")
        st.caption("Or paste JSON here:")
        st.session_state.pots_json_text = st.text_area(
            label="Pots & countries JSON",
            value=st.session_state.pots_json_text,
            height=260,
            help="Provide 'pots' plus either 'team_to_country' or 'participants_by_country'.",
        )
    else:
        if DATA_MODULE is None:
            st.warning("`data.py` not found. Create one next to draw.py with variables: pots, and team_to_country or participants_by_country.")
        else:
            st.success("Using `data.py` as input source.")

    st.session_state.forbid_same_country = st.checkbox(
        "Disallow same-country opponents",
        value=st.session_state.forbid_same_country,
        help="Uncheck if your dataset makes this too tight to solve."
    )
    st.session_state.seed = st.number_input("Random seed", value=st.session_state.seed, step=1)

    colA, colB = st.columns(2)
    with colA:
        if st.button("Start / Reset draw", use_container_width=True):
            start_or_reset()
    with colB:
        if st.button("Skip to final draw", use_container_width=True):
            if st.session_state.edges is None:
                start_or_reset()
            reveal_all()

    st.markdown("---")
    st.caption("Tip: After starting, use “Draw next team” on the main page to emulate the ceremony.")

# Show data.py preview for convenience
display_data_module_preview()

st.markdown("---")

# Bowls
pots = st.session_state.pots
edges = st.session_state.edges
orientation = st.session_state.orientation

st.subheader("Bowls")
if pots is None:
    st.info("Load your inputs and press **Start / Reset draw** to begin.")
else:
    revealed_teams = set(t for (_, t) in st.session_state.reveal_sequence[:st.session_state.reveal_pointer])
    c1, c2, c3, c4 = st.columns(4)
    with c1:
        format_bowl("Pot 1", pots[1], revealed_teams)
    with c2:
        format_bowl("Pot 2", pots[2], revealed_teams)
    with c3:
        format_bowl("Pot 3", pots[3], revealed_teams)
    with c4:
        format_bowl("Pot 4", pots[4], revealed_teams)

st.markdown("---")

# Ceremony
st.subheader("Ceremony")
if edges is None or orientation is None:
    st.info("Press **Start / Reset draw** in the sidebar to compute a valid full draw.")
else:
    col1, col2 = st.columns([1, 3])
    with col1:
        st.button("🎱 Draw next team", on_click=reveal_next, use_container_width=True)

    pot_of = {t: p for p, lst in pots.items() for t in lst}
    revealed = st.session_state.reveal_sequence[:st.session_state.reveal_pointer]
    remaining = st.session_state.reveal_sequence[st.session_state.reveal_pointer:]

    if revealed:
        st.write("### Revealed teams")
        for (p, team) in revealed:
            st.markdown(f"**Pot {p} draw:** {team}")
            team_edges = edges_for_team(team, edges)
            groups = defaultdict(list)
            for a, b in team_edges:
                opp = b if a == team else a
                venue = "Home" if is_home(team, opp, orientation) else "Away"
                groups[pot_of[opp]].append((opp, venue))

            cols = st.columns(4)
            for idx, pot_num in enumerate((1, 2, 3, 4)):
                with cols[idx]:
                    st.caption(f"Opponents from Pot {pot_num}")
                    if groups[pot_num]:
                        for opp, venue in sorted(groups[pot_num], key=lambda x: x[0]):
                            st.write(f"- {opp} — *{venue}*")
                    else:
                        st.write("—")
            st.markdown("---")
    else:
        st.info("Click **Draw next team** to reveal the first Pot 1 team and its 8 opponents.")

    if not remaining:
        st.success("🎉 Final draw complete!")
        st.write("### Per-team fixtures (8 opponents each)")
        for p in (1, 2, 3, 4):
            st.markdown(f"**Pot {p} teams**")
            for team in sorted(pots[p]):
                team_edges = edges_for_team(team, edges)
                opps = []
                for a, b in team_edges:
                    opp = b if a == team else a
                    venue = "Home" if is_home(team, opp, orientation) else "Away"
                    opps.append((pot_of[opp], opp, venue))
                opps.sort()
                nice = ", ".join([f"{opp} ({venue}, P{pot})" for pot, opp, venue in opps])
                st.write(f"- {team}: {nice}")
            st.markdown("---")

        download_links(pots, edges, orientation)

st.caption(
    "This tool simulates the ceremonial flow while guaranteeing a global valid allocation "
    "under the chosen constraints. If generation fails, loosen constraints or change the seed."
)