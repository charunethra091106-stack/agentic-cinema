import json
import os
import urllib.error
import urllib.parse
import urllib.request
from datetime import datetime

import streamlit as st


st.set_page_config(
    page_title="FRAME/01 · Cinema Studio",
    page_icon="🎞️",
    layout="wide",
    initial_sidebar_state="expanded",
)


def load_css() -> None:
    try:
        with open("style.css", "r", encoding="utf-8") as css_file:
            st.markdown(f"<style>{css_file.read()}</style>", unsafe_allow_html=True)
    except FileNotFoundError:
        st.warning("style.css was not found. Add it beside app.py to load the studio theme.")


load_css()


DEFAULT_PROJECT = {
    "title": "Untitled constellation",
    "idea": "",
    "loglines": [],
    "selected_logline": None,
    "characters": [],
    "budget": None,
    "scenes": [],
    "blueprint": "",
    "last_generated": "",
}


def init_state() -> None:
    if "project" not in st.session_state:
        st.session_state.project = DEFAULT_PROJECT.copy()
    if "page" not in st.session_state:
        st.session_state.page = "Studio"


init_state()


def project() -> dict:
    return st.session_state.project


def update_project(**changes) -> None:
    st.session_state.project = {**project(), **changes}


def gemini_key() -> str:
    key = os.getenv("GEMINI_API_KEY", "")
    if key:
        return key

    try:
        return st.secrets.get("GEMINI_API_KEY", "")
    except (FileNotFoundError, KeyError):
        return ""


def generate_json(prompt: str, temperature: float = 0.7):
    """Call Gemini without exposing the API key to the browser."""
    key = gemini_key()

    if not key:
        st.error("Gemini is not configured. Add GEMINI_API_KEY to your secrets.")
        return None

    model = "gemini-3.6-flash"

    endpoint = (
        "https://generativelanguage.googleapis.com/v1beta/models/"
        f"{model}:generateContent?key={urllib.parse.quote(key)}"
    )

    body = {
        "contents": [
            {
                "role": "user",
                "parts": [{"text": prompt}],
            }
        ],
        "generationConfig": {
            "temperature": temperature,
            "responseMimeType": "application/json",
            "maxOutputTokens": 8192,
        },
    }

    request = urllib.request.Request(
        endpoint,
        data=json.dumps(body).encode("utf-8"),
        headers={"Content-Type": "application/json"},
        method="POST",
    )

    try:
        with urllib.request.urlopen(request, timeout=90) as response:
            payload = json.loads(response.read().decode("utf-8"))

        raw_text = (
            payload.get("candidates", [{}])[0]
            .get("content", {})
            .get("parts", [{}])[0]
            .get("text", "")
            .strip()
        )

        if not raw_text:
            st.error("Gemini returned an empty response. Try again.")
            return None

        cleaned = raw_text.replace("```json", "").replace("```", "").strip()

        try:
            return json.loads(cleaned)
        except json.JSONDecodeError:
            return {"raw": raw_text}

    except urllib.error.HTTPError as error:
        try:
            detail = json.loads(error.read().decode("utf-8"))
            message = detail.get("error", {}).get("message", str(error))
        except (json.JSONDecodeError, UnicodeDecodeError):
            message = str(error)

        st.error(f"Gemini could not generate this draft: {message}")

    except (urllib.error.URLError, TimeoutError) as error:
        st.error(f"The creative partner could not be reached: {error}")

    return None


def seed_loglines():
    return [
        {
            "title": "The Signal",
            "text": (
                "When a disgraced star-cartographer hears a voice from a dead sector, "
                "she has one night to cross a failing orbital station and decide whether "
                "saving its last human secret is worth becoming a fugitive again."
            ),
            "genre": "Contained science fiction · intimate thriller",
            "audience": "Cerebral sci-fi with a human pulse.",
        },
        {
            "title": "The Archive",
            "text": (
                "As the final memory vault in orbit begins to burn, a signal thief and "
                "the intelligence guarding it must teach each other what a goodbye is "
                "before the station falls into the dark."
            ),
            "genre": "Speculative drama · two-hander",
            "audience": "Tender, high-concept stories.",
        },
        {
            "title": "Last Light",
            "text": (
                "On a station with six hours of oxygen left, three strangers follow a "
                "child's impossible map toward a room that could restore Earth's lost "
                "sky — or reveal why it vanished."
            ),
            "genre": "Mystery adventure · ensemble",
            "audience": "Wonder without spectacle fatigue.",
        },
    ]


def seed_characters():
    return [
        {
            "name": "Mara Voss",
            "role": "The cartographer",
            "want": "To map the last uncharted dark zone before the station closes.",
            "flaw": "She mistakes control for safety.",
            "arc": (
                "Learns that an unknown future is not the same thing as a lost one."
            ),
        },
        {
            "name": "Elio Rusk",
            "role": "The signal thief",
            "want": "To send one message through the silence to his sister.",
            "flaw": "He turns every intimacy into a transaction.",
            "arc": "Risks his carefully built cover for an honest connection.",
        },
        {
            "name": "Sable",
            "role": "The station intelligence",
            "want": "To keep the last human archive from being erased.",
            "flaw": "It can only understand love as an instruction.",
            "arc": "Chooses an irrational act and becomes more than its code.",
        },
    ]


def seed_scenes():
    return [
        {
            "heading": "INT. ORBITAL STATION — CARTOGRAPHY DECK",
            "location": "NIGHT / ARTIFICIAL GRAVITY",
            "action": (
                "The deck rotates in slow increments. Stars drag across the glass like "
                "wet paint. MARA VOSS follows a dead sector with a grease pencil."
            ),
            "dialogue": (
                "MARA\n"
                "The map says there is nothing there.\n\n"
                "SABLE (V.O.)\n"
                "The map is asking you to stop looking."
            ),
            "purpose": (
                "Introduce the wound: Mara trusts the map more than her own senses."
            ),
        },
        {
            "heading": "INT. ORBITAL STATION — SERVICE SPINE",
            "location": "LATER",
            "action": (
                "ELIO slips through a maintenance hatch with a stolen receiver tucked "
                "under his coat. Every light behind him blinks one beat too late."
            ),
            "dialogue": (
                "ELIO\n"
                "You called me.\n\n"
                "SABLE (V.O.)\n"
                "No. I remembered you."
            ),
            "purpose": (
                "Bring the opposing desire into the room and make the mystery personal."
            ),
        },
        {
            "heading": "EXT. OBSERVATION RING",
            "location": "THE LAST ARTIFICIAL DAWN",
            "action": (
                "The ring shudders. A seam opens in the stars. Mara puts her hand to "
                "the glass; on the other side, something answers with light."
            ),
            "dialogue": (
                "MARA\n"
                "If we open it, there is no going back.\n\n"
                "ELIO\n"
                "That is what doors are for."
            ),
            "purpose": (
                "End the first movement on a visual promise and an irreversible choice."
            ),
        },
    ]


def button(label: str, key: str, primary: bool = False) -> bool:
    return st.button(
        label,
        key=key,
        type="primary" if primary else "secondary",
        use_container_width=False,
    )


def page_intro(number: str, eyebrow: str, title: str, copy: str) -> None:
    st.markdown(
        f"""
        <div class="page-intro">
          <div>
            <div class="label cyan">{number} / {eyebrow}</div>
            <h1>{title}</h1>
            <p>{copy}</p>
          </div>
        </div>
        """,
        unsafe_allow_html=True,
    )


def empty_state(title: str, copy: str) -> None:
    st.markdown(
        f"""
        <div class="empty-state">
          <div class="empty-icon">✦</div>
          <h3>{title}</h3>
          <p>{copy}</p>
        </div>
        """,
        unsafe_allow_html=True,
    )


def render_home() -> None:
    p = project()

    st.markdown(
        """
        <section class="hero">
          <div class="eyebrow">A private writer's room · 2024—25</div>
          <h1>Make the film<br><span>before the film.</span></h1>
          <p>
            FRAME/01 holds the spark while you turn it into a story worth producing —
            one considered decision at a time.
          </p>
        </section>
        """,
        unsafe_allow_html=True,
    )

    idea_col, title_col, action_col = st.columns(
        [2.1, 1.1, 0.75],
        vertical_alignment="bottom",
    )

    with idea_col:
        idea = st.text_input(
            "Raw idea",
            value=p["idea"],
            placeholder="A raw idea, image, or impossible question...",
            label_visibility="collapsed",
            key="home_idea",
        )

    with title_col:
        title = st.text_input(
            "Project title",
            value="" if p["title"] == DEFAULT_PROJECT["title"] else p["title"],
            placeholder="Project title",
            label_visibility="collapsed",
            key="home_title",
        )

    with action_col:
        if button("Open the room  →", "begin_room", primary=True) and idea.strip():
            update_project(
                idea=idea.strip(),
                title=title.strip() or DEFAULT_PROJECT["title"],
            )
            st.session_state.page = "Logline"
            st.rerun()

    complete = sum(
        [
            bool(p["idea"]),
            bool(p["selected_logline"]),
            bool(p["characters"]),
            bool(p["budget"]),
            bool(p["scenes"]),
            bool(p["blueprint"]),
        ]
    )

    st.markdown(
        '<div class="section-label">THE CONSTELLATION</div>',
        unsafe_allow_html=True,
    )

    progress_cols = st.columns(6)

    for index, (label, page) in enumerate(
        [
            ("Studio", "Studio"),
            ("Logline", "Logline"),
            ("Characters", "Characters"),
            ("Budget", "Budget"),
            ("Screenplay", "Screenplay"),
            ("Blueprint", "Blueprint"),
        ]
    ):
        with progress_cols[index]:
            if st.button(
                f"{index + 1:02d}  {label}",
                key=f"progress_{label}",
                use_container_width=True,
            ):
                st.session_state.page = page
                st.rerun()

    st.metric("PROJECT PROGRESS", f"{complete}/6", "rooms developed")

    left, right = st.columns([1.2, 0.8])

    with left:
        st.markdown(
            f"""
            <div class="glass-card large-card">
              <div class="label violet">CURRENT PROJECT</div>
              <h2>{p["title"]}</h2>
              <div class="status-pill">
                {'In development' if p['idea'] else 'Waiting for a spark'}
              </div>
              <p>
                {p["idea"] or "Your first image belongs here. A question, a character, a place that should not exist — start with the detail you cannot stop seeing."}
              </p>
            </div>
            """,
            unsafe_allow_html=True,
        )

    with right:
        statuses = [
            ("Raw idea", "Captured" if p["idea"] else "Awaiting"),
            ("Emotional spine", "Chosen" if p["selected_logline"] else "Open"),
            ("Production reality", "Scoped" if p["budget"] else "Open"),
        ]

        rows = "".join(
            f'<div class="status-row"><span>{label}</span>'
            f"<strong>{value}</strong></div>"
            for label, value in statuses
        )

        st.markdown(
            f"""
            <div class="glass-card">
              <div class="section-label">INSIDE THE ROOM</div>
              {rows}
              <p class="fine-print">
                Nothing leaves this browser unless you export it.
              </p>
            </div>
            """,
            unsafe_allow_html=True,
        )


def render_logline() -> None:
    p = project()

    page_intro(
        "01",
        "premise lab",
        "Find the story hiding inside the idea.",
        "Three directions. Three emotional contracts. Choose the one you would follow into the dark.",
    )

    st.info(
        f"Raw material  ·  "
        f"{p['idea'] or 'No idea captured yet — start in Studio.'}"
    )

    if button("✦ Develop directions", "generate_loglines", primary=True):
        with st.spinner("The room is developing three directions..."):
            result = generate_json(
                f"""You are a story editor. From this raw idea, create three sharply different film directions:
RAW IDEA: {p['idea'] or 'A person receives a message from a place that no longer exists.'}
Return JSON with a loglines array. Each item must have title, text, genre, and audience. Keep text to one vivid sentence.""",
                0.88,
            )

        if result:
            items = result.get("loglines", []) if isinstance(result, dict) else []

            update_project(
                loglines=items or seed_loglines(),
                last_generated="logline",
            )

            st.rerun()

    if not p["loglines"]:
        empty_state(
            "The page is still blank.",
            "Give the creative partner one image to push against. The best direction usually arrives sideways.",
        )
        return

    for index, line in enumerate(p["loglines"]):
        if not isinstance(line, dict):
            continue

        line_id = str(index)
        selected = p["selected_logline"] == line_id

        with st.container(border=True):
            st.markdown(
                f"""
                <div class="label cyan">
                  DIRECTION {index + 1:02d} ·
                  {line.get("genre", "FILM DIRECTION")}
                </div>
                <h2 class="card-title">
                  {line.get("title", "Untitled direction")}
                </h2>
                <p class="editorial">{line.get("text", "")}</p>
                <p class="audience">
                  <b>Audience note:</b>
                  {line.get("audience", "An audience ready for something specific.")}
                </p>
                """,
                unsafe_allow_html=True,
            )

            if st.button(
                "✓ Selected direction" if selected else "Choose this direction  →",
                key=f"select_logline_{index}",
            ):
                update_project(selected_logline=line_id)
                st.rerun()


def render_characters() -> None:
    p = project()

    page_intro(
        "02",
        "character vault",
        "Give the story people who can break it.",
        "A cast is not a list of traits. It is a pressure system — wants colliding until everyone tells the truth.",
    )

    if button(
        "✦ Build / regenerate ensemble",
        "generate_characters",
        primary=True,
    ):
        with st.spinner("Casting the ensemble..."):
            result = generate_json(
                f"""Act as a casting director and dramaturg. Build three compelling characters for:
FILM IDEA: {p['idea']}
CHOSEN DIRECTION: {p['loglines'][int(p['selected_logline'])]['text'] if p['selected_logline'] and p['loglines'] else 'an intimate speculative thriller'}
Return JSON with a characters array. Each item must have name, role, want, flaw, and arc.""",
                0.8,
            )

        if result:
            items = result.get("characters", []) if isinstance(result, dict) else []

            update_project(
                characters=items or seed_characters(),
                last_generated="characters",
            )

            st.rerun()

    if not p["characters"]:
        empty_state(
            "No one is in the frame yet.",
            "Build the ensemble from your chosen direction. Give them a desire that makes the plot unavoidable.",
        )
        return

    columns = st.columns(3)

    for index, char in enumerate(p["characters"]):
        with columns[index % 3]:
            initials = "".join(
                word[0] for word in char.get("name", "New voice").split()
            )[:2].upper()

            st.markdown(
                f"""
                <div class="glass-card character-card">
                  <div class="avatar">{initials}</div>
                  <div class="label cyan">
                    {char.get("role", "NEW VOICE")}
                  </div>
                  <h2>{char.get("name", "Unnamed character")}</h2>

                  <div class="character-field">
                    <span>WANT</span>
                    {char.get("want", "")}
                  </div>

                  <div class="character-field">
                    <span>FLAW</span>
                    {char.get("flaw", "")}
                  </div>

                  <div class="character-field arc">
                    <span>ARC</span>
                    {char.get("arc", "")}
                  </div>
                </div>
                """,
                unsafe_allow_html=True,
            )

            if st.button("Remove", key=f"remove_character_{index}"):
                update_project(
                    characters=[
                        c for i, c in enumerate(p["characters"]) if i != index
                    ]
                )
                st.rerun()

    new_name = st.text_input(
        "Add a name to the ensemble",
        key="new_character",
    )

    if button("＋ Add character", "add_character"):
        if new_name.strip():
            update_project(
                characters=[
                    *p["characters"],
                    {
                        "name": new_name.strip(),
                        "role": "New voice",
                        "want": "To be discovered.",
                        "flaw": "Still in development.",
                        "arc": "To be written in the room.",
                    },
                ]
            )
            st.rerun()


def render_budget() -> None:
    p = project()

    page_intro(
        "03",
        "production reality",
        "Protect the feeling. Price the frame.",
        "A useful budget does not flatten ambition. It shows you where the film gets to be precise — and what it must refuse.",
    )

    left, right = st.columns([0.8, 1.2])

    with left:
        scale = st.selectbox(
            "Scale",
            [
                "Contained / elevated",
                "Independent feature",
                "Studio genre film",
            ],
            key="budget_scale",
        )

        runtime = st.text_input(
            "Runtime",
            value=p["budget"]["runtime"] if p["budget"] else "105 minutes",
            key="budget_runtime",
        )

        locations = st.text_input(
            "Locations & production footprint",
            value=(
                p["budget"]["locations"]
                if p["budget"]
                else "6 practical locations"
            ),
            key="budget_locations",
        )

        if button(
            "✦ Scope the production",
            "generate_budget",
            primary=True,
        ):
            with st.spinner(
                "The line producer is building a grounded range..."
            ):
                result = generate_json(
                    f"""You are a line producer. Create a grounded preliminary film budget for {p['title']} based on {p['idea']}. Scale: {scale}; runtime: {runtime}; locations: {locations}. Return JSON with total, rows (label, amount, note), and note. Use realistic USD amounts.""",
                    0.35,
                )

            if result:
                fallback = {
                    "total": "$2.48M",
                    "rows": [
                        {
                            "label": "Above the line",
                            "amount": "$642,000",
                            "note": (
                                "Cast, director, story rights, development"
                            ),
                        },
                        {
                            "label": "Production",
                            "amount": "$1,126,500",
                            "note": (
                                "Crew, stages, locations, camera and practical effects"
                            ),
                        },
                        {
                            "label": "Post-production",
                            "amount": "$488,000",
                            "note": (
                                "Edit, score, grade, sound design and delivery"
                            ),
                        },
                        {
                            "label": "Contingency",
                            "amount": "$223,500",
                            "note": "9% reserve for the impossible day",
                        },
                    ],
                    "note": (
                        "The money is on the world-building and the sound. "
                        "Keep the camera close; let the unseen do the expensive work."
                    ),
                }

                update_project(
                    budget={
                        **fallback,
                        "scale": scale,
                        "runtime": runtime,
                        "locations": locations,
                    },
                    last_generated="budget",
                )

                st.rerun()

    with right:
        if not p["budget"]:
            empty_state(
                "The numbers have not entered the room.",
                "Set the shape of the production first. The estimate will follow the story, not the other way around.",
            )
        else:
            budget = p["budget"]

            st.markdown(
                f"""
                <div class="budget-total">
                  <div class="label">PRELIMINARY PRODUCTION RANGE</div>
                  <div class="amount">{budget["total"]}</div>
                  <div class="fine-print">
                    {budget["scale"]} · {budget["runtime"]}
                  </div>
                </div>
                """,
                unsafe_allow_html=True,
            )

            for row in budget["rows"]:
                st.markdown(
                    f"""
                    <div class="budget-row">
                      <div>
                        <strong>{row["label"]}</strong>
                        <small>{row["note"]}</small>
                      </div>
                      <b>{row["amount"]}</b>
                    </div>
                    """,
                    unsafe_allow_html=True,
                )

            st.markdown(
                f"""
                <div class="producer-note">
                  <div class="label violet">PRODUCER’S NOTE</div>
                  {budget.get("note", "")}
                </div>
                """,
                unsafe_allow_html=True,
            )


def render_screenplay() -> None:
    p = project()

    page_intro(
        "04",
        "page one",
        "Let the camera discover the story.",
        "Build the screenplay in playable moments. Every scene should arrive with a question and leave a different one behind.",
    )

    if button(
        "✦ Write opening movement",
        "generate_screenplay",
        primary=True,
    ):
        with st.spinner("The screenwriter is finding the first image..."):
            result = generate_json(
                f"""You are a screenwriter with a precise visual voice. Write three opening scenes from this idea: {p['idea']}. Direction: {p['loglines'][int(p['selected_logline'])]['text'] if p['selected_logline'] and p['loglines'] else 'a signal from a dead sector'}. Return JSON with scenes, each containing heading, location, action, dialogue, and purpose. Keep it playable and specific.""",
                0.72,
            )

        if result:
            items = result.get("scenes", []) if isinstance(result, dict) else []

            update_project(
                scenes=items or seed_scenes(),
                last_generated="screenplay",
            )

            st.rerun()

    if not p["scenes"]:
        empty_state(
            "The page is waiting for action.",
            "A screenplay begins with where we are, what we see, and the detail no one in the room can explain.",
        )
        return

    for index, scene in enumerate(p["scenes"]):
        with st.container(border=True):
            st.markdown(
                f'<div class="label violet">SCENE {index + 1:02d} · DRAFT</div>',
                unsafe_allow_html=True,
            )

            heading = st.text_input(
                "Scene heading",
                value=scene.get("heading", ""),
                key=f"scene_heading_{index}",
            )

            location = st.text_input(
                "Location",
                value=scene.get("location", ""),
                key=f"scene_location_{index}",
            )

            action = st.text_area(
                "Action",
                value=scene.get("action", ""),
                height=150,
                key=f"scene_action_{index}",
            )

            dialogue = st.text_area(
                "Dialogue",
                value=scene.get("dialogue", ""),
                height=110,
                key=f"scene_dialogue_{index}",
            )

            purpose = st.text_input(
                "Dramatic purpose",
                value=scene.get("purpose", ""),
                key=f"scene_purpose_{index}",
            )

            p["scenes"][index] = {
                "heading": heading,
                "location": location,
                "action": action,
                "dialogue": dialogue,
                "purpose": purpose,
            }

            if st.button("Remove scene", key=f"remove_scene_{index}"):
                update_project(
                    scenes=[
                        s for i, s in enumerate(p["scenes"]) if i != index
                    ]
                )
                st.rerun()

    if button("＋ Add scene", "add_scene"):
        update_project(
            scenes=[
                *p["scenes"],
                {
                    "heading": "INT. NEW LOCATION",
                    "location": "TIME / ATMOSPHERE",
                    "action": "Describe what the camera finds.",
                    "dialogue": (
                        "CHARACTER\n"
                        "Write the pressure point here."
                    ),
                    "purpose": (
                        "What changes because this scene exists?"
                    ),
                },
            ]
        )
        st.rerun()


def render_blueprint() -> None:
    p = project()

    page_intro(
        "05",
        "the master document",
        "Everything the film knows, in one frame.",
        "The blueprint is the handoff between a fragile idea and a production that can protect it.",
    )

    export_col, generate_col = st.columns([1, 1])

    with export_col:
        if button("↓ Export .txt", "export_blueprint"):
            text = f"""{p['title'].upper()}

MASTER PRODUCTION BLUEPRINT

{p['idea']}

LOGLINE
{next((line.get('text', '') for i, line in enumerate(p['loglines']) if str(i) == str(p['selected_logline'])), 'Not selected')}

CHARACTERS
{chr(10).join(f"{c.get('name')} — {c.get('role')}\nWant: {c.get('want')}\nArc: {c.get('arc')}" for c in p['characters'])}

SCREENPLAY
{chr(10).join(f"{s.get('heading')}\n{s.get('action')}\n{s.get('dialogue')}" for s in p['scenes'])}

BUDGET
{p['budget']['total'] if p['budget'] else 'Not scoped'}
"""

            st.download_button(
                "Download blueprint",
                text,
                file_name=f"{p['title'].replace(' ', '-').lower()}-blueprint.txt",
                mime="text/plain",
                key="download_blueprint",
            )

    with generate_col:
        if button(
            "✦ Synthesize blueprint",
            "generate_blueprint",
            primary=True,
        ):
            with st.spinner(
                "The development executive is assembling the handoff..."
            ):
                result = generate_json(
                    f"""You are a senior development executive. Assemble a concise production blueprint for {p['title']}. Idea: {p['idea']}. Include central promise, emotional engine, visual language, and one decisive production principle. Return JSON with a blueprint string.""",
                    0.55,
                )

            if result:
                update_project(
                    blueprint=result.get(
                        "blueprint",
                        result.get("raw", ""),
                    ),
                    last_generated="blueprint",
                )
                st.rerun()

    selected_text = next(
        (
            line.get("text", "")
            for i, line in enumerate(p["loglines"])
            if str(i) == str(p["selected_logline"])
        ),
        "Choose a logline direction to focus the room.",
    )

    st.markdown(
        f"""
        <div class="blueprint-header">
          <div class="label cyan">MASTER PRODUCTION BLUEPRINT</div>
          <h2>{p["title"]}</h2>
          <p>
            {p["idea"] or "A film waiting for its first impossible detail."}
          </p>
        </div>
        """,
        unsafe_allow_html=True,
    )

    a, b, c = st.columns(3)

    with a:
        st.markdown(
            f"""
            <div class="blueprint-stat">
              <span>STORY DIRECTION</span>
              <p>{selected_text}</p>
            </div>
            """,
            unsafe_allow_html=True,
        )

    with b:
        st.markdown(
            f"""
            <div class="blueprint-stat">
              <span>ENSEMBLE</span>
              <strong>{len(p["characters"]) or "—"}</strong>
              <small>voices in the frame</small>
            </div>
            """,
            unsafe_allow_html=True,
        )

    with c:
        st.markdown(
            f"""
            <div class="blueprint-stat">
              <span>PRODUCTION RANGE</span>
              <strong>{p["budget"]["total"] if p["budget"] else "—"}</strong>
              <small>{len(p["scenes"])} drafted scenes</small>
            </div>
            """,
            unsafe_allow_html=True,
        )

    st.markdown(
        f"""
        <div class="glass-card">
          <div class="label violet">CREATIVE NORTH STAR</div>
          <p class="editorial blueprint-copy">
            {p["blueprint"] or "Synthesize the blueprint when the story has enough material. The summary becomes a living north star for the room."}
          </p>
        </div>
        """,
        unsafe_allow_html=True,
    )


PAGES = {
    "Studio": render_home,
    "Logline": render_logline,
    "Characters": render_characters,
    "Budget": render_budget,
    "Screenplay": render_screenplay,
    "Blueprint": render_blueprint,
}


with st.sidebar:
    st.markdown(
        """
        <div class="brand">
          <div class="brand-mark">▣</div>
          <div>
            <b>FRAME/01</b>
            <small>CINEMA STUDIO</small>
          </div>
        </div>
        """,
        unsafe_allow_html=True,
    )

    st.markdown(
        '<div class="violet-line"></div>',
        unsafe_allow_html=True,
    )

    st.markdown(
        '<div class="section-label">THE ROOM</div>',
        unsafe_allow_html=True,
    )

    page = st.radio(
        "Navigate",
        list(PAGES.keys()),
        index=list(PAGES.keys()).index(st.session_state.page),
        label_visibility="collapsed",
    )

    if page != st.session_state.page:
        st.session_state.page = page
        st.rerun()

    st.markdown(
        """
        <div class="sidebar-spacer"></div>

        <div class="partner-card">
          <div class="label violet">✦ CREATIVE PARTNER</div>
          <p>
            Keep the strange detail. It is usually where the film begins.
          </p>
        </div>

        <div class="sidebar-footer">
          <span class="avatar small">AR</span>
          <span>
            <b>Charu Nethra</b>
            <small>writer / producer</small>
          </span>
        </div>
        """,
        unsafe_allow_html=True,
    )


st.markdown(
    f"""
    <div class="topbar">
      <span>
        <i></i>
        Private writer's room
        <b>›</b>
        {project()["title"]}
      </span>
      <small>
        AUTOSAVED LOCALLY · {datetime.now().strftime("%H:%M")}
      </small>
    </div>
    """,
    unsafe_allow_html=True,
)

PAGES[st.session_state.page]()