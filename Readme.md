# UEFA Champions League — League Phase Draw Simulator

Interactive Streamlit app to simulate the 36-team UEFA Champions League **league-phase** draw.
It mirrors the ceremony: balls drawn from four pots, opponents allocated by (virtual) software, and home/away revealed — with an option to **skip straight to the final draw**.

> ⚠️ This is a fan-made simulator. It's not affiliated with UEFA.

---

## Features

* **36 teams, 4 pots (9 each)**.
* For every club, exactly **8 opponents**: **2 from each pot** (including its own pot).
* **Per-pot home/away balance**: vs each pot, a team gets **1 home + 1 away** (so 4 home + 4 away in total).
* **No duplicate pairings**.
* **Same-country restriction** toggle (on by default).
* **Two input sources** (JSON takes priority when selected):

  * Paste or upload **JSON** in the UI.
  * Load from local **`data.py`** (module next to `draw.py`).
* **Nice preview of `data.py`** (pots, team→country, participants by country).
* **Step-by-step ceremony** (Pot 1 → Pot 4) or **Skip to final draw**.
* **Download** the resulting fixtures as JSON.

---

## Quick Start

### Requirements

* Python 3.9+
* `streamlit` (and only standard library otherwise)

```bash
pip install streamlit
```

### Run

```bash
streamlit run draw.py
```

Open the local URL Streamlit prints (usually [http://localhost:8501](http://localhost:8501)).

---

## Input Data

You can use **JSON** or a local **`data.py`** file. Select the source in the sidebar.
When **JSON** is selected, it has priority over `data.py`.

### A) JSON (recommended)

Provide `pots` and either `team_to_country` **or** `participants_by_country`.

```json
{
  "pots": {
    "1": ["Team A1", "Team A2", "...", "Team A9"],
    "2": ["Team B1", "...", "Team B9"],
    "3": ["Team C1", "...", "Team C9"],
    "4": ["Team D1", "...", "Team D9"]
  },
  "team_to_country": {
    "Team A1": "Country X",
    "Team A2": "Country X",
    "Team B1": "Country Y"
    /* ... all 36 teams mapped ... */
  }
}
```

*or*

```json
{
  "pots": { "1": [...], "2": [...], "3": [...], "4": [...] },
  "participants_by_country": {
    "Country X": ["Team A1", "Team A2"],
    "Country Y": ["Team B1"]
    /* ... every team included somewhere ... */
  }
}
```

**Rules for JSON:**

* Exactly **4 pots**, each with **9 unique** team names.
* No team appears in more than one pot.
* Every team in `pots` must be present in either `team_to_country` or `participants_by_country`.

### B) `data.py` (module)

Create a `data.py` alongside `draw.py` with **any** of the following variables:

```python
# data.py
pots = {
    "1": ["Team A1", "...", "Team A9"],
    "2": ["Team B1", "...", "Team B9"],
    "3": ["Team C1", "...", "Team C9"],
    "4": ["Team D1", "...", "Team D9"],
}

# Prefer team_to_country if you have it:
team_to_country = {
    "Team A1": "Country X",
    "Team B1": "Country Y",
    # ...all 36...
}

# Or use participants_by_country instead:
# participants_by_country = {
#     "Country X": ["Team A1", "Team A2"],
#     "Country Y": ["Team B1"],
#     # ...every team included...
# }
```

The app shows a **preview** of what it read from `data.py` (pots, team→country, and participants by country).

---

## Using the App

1. **Choose input source** (JSON or `data.py`) in the sidebar.
2. If JSON:

   * Upload a file **or** paste JSON into the text area.
3. (Optional) Toggle **"Disallow same-country opponents"**.
4. (Optional) Adjust the **Random seed** for reproducibility.
5. Click **"Start / Reset draw"**.
6. Use **"🎱 Draw next team"** to reveal the ceremony team-by-team (Pot 1 → Pot 4),
   or click **"Skip to final draw"** to reveal everything at once.
7. Click **"⬇️ Download draw (JSON)"** to export fixtures.

**Bowls** at the top show remaining balls per pot as the ceremony progresses.

---

## Rules & Constraints (what the simulator enforces)

* 36 teams separated into **four pots of nine**.
* Each team is paired with **8 unique opponents**:

  * **2 from Pot 1**, **2 from Pot 2**, **2 from Pot 3**, **2 from Pot 4**.
* **No duplicates** (a pair of teams meets once).
* **Per-pot Home/Away balance** (strict):
  For every team and every pot, it will have **exactly one Home** match and **one Away** match.
  ⇒ total **4 Home + 4 Away** per team.
* **Same country** restrictions are **toggleable**:

  * **ON (default)**: teams from the same association will not be paired.
  * **OFF**: simulator will ignore country clashes (useful for constrained datasets).

---

## How It Works (in brief)

1. **Edges generation (opponent selection)**

   * Builds all 144 pairings using a randomized greedy + local backtracking:

     * 9 edges within each pot.
     * 18 edges for each of the six cross-pot pairs.
   * Respects "need 2 opponents from each pot" per team and optional same-country rule.
   * Restarts with a new randomization if no full solution is found within limits.

2. **Home/Away assignment**

   * Orients every pairing so each team gets **1 home + 1 away vs each pot**.
   * Retries with different orders until a feasible orientation is found.

3. **Ceremony layer**

   * Once a full valid allocation exists, the UI simply **reveals** it in pot order, team by team, emulating the broadcast.

---

## Troubleshooting

* **"Failed to generate a valid draw…"**

  * Try a different **Random seed** (sidebar).
  * **Uncheck** "Disallow same-country opponents" if your dataset is too tight.
  * Verify your JSON/`data.py` has **exactly 9 teams per pot** and **no duplicates**.

* **"Input error: …"**

  * Ensure every team in `pots` is mapped to a country (via `team_to_country` or `participants_by_country`).
  * Check for typos, missing commas, or inconsistent casing in team names.

* **`data.py` not found**

  * Place `data.py` in the same folder as `draw.py`.
  * The app tells you if import fails and shows the reason.

---

## Customization Tips

* **Tighten/loosen search**

  * In code, `max_global_tries` (edge generation) and `max_tries` (home/away) control how long the solver looks for a solution.

* **Add extra constraints**

  * The opponent selection step is centralized in `build_full_draw(...)`.
  * You can add rules (e.g., time-zone groups, stadium clashes) by expanding `compatible(...)`.

* **Reproducibility**

  * Set a fixed **Random seed** to reproduce the same draw.

---

## Project Structure

```
.
├─ draw.py        # Streamlit app
├─ data.py        # (optional) Teams & pots source for the app
└─ README.md
```

---

## Sample JSON (copy/paste into the app)

```json
{
  "pots": {
    "1": ["Sample Team P1-01","Sample Team P1-02","Sample Team P1-03","Sample Team P1-04","Sample Team P1-05","Sample Team P1-06","Sample Team P1-07","Sample Team P1-08","Sample Team P1-09"],
    "2": ["Sample Team P2-01","Sample Team P2-02","Sample Team P2-03","Sample Team P2-04","Sample Team P2-05","Sample Team P2-06","Sample Team P2-07","Sample Team P2-08","Sample Team P2-09"],
    "3": ["Sample Team P3-01","Sample Team P3-02","Sample Team P3-03","Sample Team P3-04","Sample Team P3-05","Sample Team P3-06","Sample Team P3-07","Sample Team P3-08","Sample Team P3-09"],
    "4": ["Sample Team P4-01","Sample Team P4-02","Sample Team P4-03","Sample Team P4-04","Sample Team P4-05","Sample Team P4-06","Sample Team P4-07","Sample Team P4-08","Sample Team P4-09"]
  },
  "team_to_country": {
    "Sample Team P1-01": "Country A",
    "Sample Team P1-02": "Country B"
    /* ...map all 36 teams... */
  }
}
```

---

## License

Add your preferred license here (e.g., MIT).

---

### Credits

Built to help fans and analysts play with the new league-phase draw format. Contributions and suggestions welcome!
