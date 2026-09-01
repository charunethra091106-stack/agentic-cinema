import os
import json
import html
import re
import time
from pathlib import Path

import streamlit as st
from dotenv import load_dotenv
from google import genai
from google.genai import types


# =========================================================
# CONFIG
# =========================================================

load_dotenv()

st.set_page_config(
    page_title="Agentic AI Cinema Studio",
    page_icon="🎬",
    layout="wide",
    initial_sidebar_state="collapsed",
)

BASE_DIR = Path(__file__).resolve().parent

MODEL_NAME = os.getenv(
    "GEMINI_MODEL",
    "gemini-2.5-flash",
)

VIDEO_URL = os.getenv(
    "SPACE_VIDEO_URL",
    "https://assets.mixkit.co/videos/preview/mixkit-stars-in-space-background-1610-large.mp4",
)

API_KEY = os.getenv("GEMINI_API_KEY")


# =========================================================
# GEMINI CLIENT
# =========================================================

client = None

if API_KEY:
    try:
        client = genai.Client(api_key=API_KEY)
    except Exception as exc:
        client = None
        st.warning(f"Gemini client could not be initialized: {exc}")


# =========================================================
# CUSTOM CSS
# =========================================================

style_path = BASE_DIR / "style.css"

if style_path.exists():
    try:
        css_content = style_path.read_text(encoding="utf-8")

        st.markdown(
            f"<style>{css_content}</style>",
            unsafe_allow_html=True,
        )
    except Exception as exc:
        st.warning(f"Could not load style.css: {exc}")


# =========================================================
# LIVE SPACE BACKGROUND
# =========================================================

st.markdown(
    f"""
    <div class="space-background">
        <video autoplay muted loop playsinline>
            <source src="{html.escape(VIDEO_URL)}" type="video/mp4">
        </video>
        <div class="space-overlay"></div>
    </div>

    <div class="stars-layer"></div>
    """,
    unsafe_allow_html=True,
)


# =========================================================
# IMAGE HELPERS
# =========================================================

def local_image(*parts):
    """Return a local image path if it exists."""
    try:
        path = BASE_DIR.joinpath(*parts)

        if path.exists() and path.is_file():
            return str(path)

    except Exception:
        pass

    return None


IMG_LOGLINE = local_image(
    "img_astro",
    "logline.jpeg",
)

IMG_CHARACTER = local_image(
    "img_romance",
    "character.jpeg",
)

IMG_SCREENPLAY = local_image(
    "img_screenplay",
    "screenplay.jpeg",
)

IMG_BUDGET = local_image(
    "img_thriller",
    "budget.jpeg",
)


# =========================================================
# SESSION STATE
# =========================================================

DEFAULTS = {
    "page": "Home",
    "logline_options": [],
    "selected_logline": "",
    "characters": [],
    "budget_result": None,
    "screenplay_result": None,
    "blueprint_result": None,
    "project_genre": "Sci-Fi",
    "project_tone": "Cinematic",
}

for key, value in DEFAULTS.items():
    if key not in st.session_state:
        st.session_state[key] = value


# =========================================================
# NAVIGATION
# =========================================================

def go(page):
    st.session_state.page = page
    st.rerun()


# =========================================================
# GENERAL HELPERS
# =========================================================

def clean(value):
    """
    Safely convert any value into HTML-safe text.
    """
    if value is None:
        return ""

    if isinstance(value, (list, tuple)):
        value = ", ".join(str(x) for x in value)

    if isinstance(value, dict):
        value = json.dumps(
            value,
            ensure_ascii=False,
        )

    return html.escape(str(value))


def safe_list(value):
    """
    Always return a list.
    Handles Gemini returning:
    - list
    - tuple
    - string
    - None
    """
    if value is None:
        return []

    if isinstance(value, list):
        return value

    if isinstance(value, tuple):
        return list(value)

    if isinstance(value, str):
        value = value.strip()

        if not value:
            return []

        return [value]

    return [str(value)]


def safe_dict(value):
    """
    Always return a dictionary.
    """
    if isinstance(value, dict):
        return value

    return {}


def as_text(value, default=""):
    """
    Convert arbitrary Gemini output into safe plain text.
    """
    if value is None:
        return default

    if isinstance(value, str):
        return value.strip()

    if isinstance(value, list):
        return ", ".join(str(x) for x in value)

    if isinstance(value, dict):
        return json.dumps(
            value,
            ensure_ascii=False,
        )

    return str(value)


# =========================================================
# AI HELPERS
# =========================================================

def require_ai():
    """
    Check whether Gemini is configured.
    """
    if not API_KEY:
        st.error(
            "Gemini API key is not configured. "
            "Add GEMINI_API_KEY=your_key to the .env file "
            "and restart Streamlit."
        )
        return False

    if client is None:
        st.error(
            "Gemini client could not be initialized. "
            "Check your API key and google-genai installation."
        )
        return False

    return True


def extract_json(text):
    """
    Robustly extract JSON from Gemini output.

    Handles:
    - normal JSON
    - ```json ... ```
    - ``` ... ```
    - accidental explanatory text around JSON
    """
    if not text:
        raise ValueError("Gemini returned an empty response.")

    text = str(text).strip()

    # Remove markdown code fences.
    text = re.sub(
        r"^```(?:json)?\s*",
        "",
        text,
        flags=re.IGNORECASE,
    )

    text = re.sub(
        r"\s*```$",
        "",
        text,
    )

    text = text.strip()

    # First attempt: direct JSON.
    try:
        return json.loads(text)
    except json.JSONDecodeError:
        pass

    # Try extracting an object.
    object_start = text.find("{")
    object_end = text.rfind("}")

    if object_start != -1 and object_end > object_start:
        candidate = text[
            object_start : object_end + 1
        ]

        try:
            return json.loads(candidate)
        except json.JSONDecodeError:
            pass

    # Try extracting an array.
    array_start = text.find("[")
    array_end = text.rfind("]")

    if array_start != -1 and array_end > array_start:
        candidate = text[
            array_start : array_end + 1
        ]

        try:
            return json.loads(candidate)
        except json.JSONDecodeError:
            pass

    raise ValueError(
        "Gemini returned invalid JSON."
    )


def ai_json(
    prompt,
    temperature=0.8,
    retries=3,
):
    """
    Call Gemini and return parsed JSON.

    Includes retry handling for temporary API failures.
    """

    if not require_ai():
        return None

    last_error = None

    for attempt in range(retries):
        try:
            response = client.models.generate_content(
                model=MODEL_NAME,
                contents=prompt,
                config=types.GenerateContentConfig(
                    temperature=temperature,
                    response_mime_type="application/json",
                ),
            )

            response_text = getattr(
                response,
                "text",
                None,
            )

            if not response_text:
                raise ValueError(
                    "Gemini returned no text."
                )

            result = extract_json(
                response_text
            )

            if not isinstance(result, dict):
                raise ValueError(
                    "Gemini JSON response must be an object."
                )

            return result

        except Exception as exc:
            last_error = exc

            # Retry temporary failures.
            if attempt < retries - 1:
                time.sleep(1.5 * (attempt + 1))

    st.error(
        f"AI generation failed after {retries} attempts: "
        f"{last_error}"
    )

    return None


def ai_text(
    prompt,
    temperature=0.8,
    retries=3,
):
    """
    Call Gemini and return normal text.
    """

    if not require_ai():
        return None

    last_error = None

    for attempt in range(retries):
        try:
            response = client.models.generate_content(
                model=MODEL_NAME,
                contents=prompt,
                config=types.GenerateContentConfig(
                    temperature=temperature,
                ),
            )

            response_text = getattr(
                response,
                "text",
                None,
            )

            if not response_text:
                raise ValueError(
                    "Gemini returned no text."
                )

            return response_text.strip()

        except Exception as exc:
            last_error = exc

            if attempt < retries - 1:
                time.sleep(1.5 * (attempt + 1))

    st.error(
        f"AI generation failed after {retries} attempts: "
        f"{last_error}"
    )

    return None


# =========================================================
# DISPLAY HELPERS
# =========================================================

def show_image(image_path, fallback_text):
    if image_path:
        try:
            st.image(
                image_path,
                use_container_width=True,
            )
            return
        except Exception:
            pass

    st.markdown(
        f"""
        <div class="image-placeholder">
            <span>🎬</span>
            <p>{clean(fallback_text)}</p>
        </div>
        """,
        unsafe_allow_html=True,
    )


def card(
    title,
    content,
    accent="cyan",
):
    st.markdown(
        f"""
        <div class="result-card accent-{clean(accent)}">
            <div class="result-card-title">
                {clean(title)}
            </div>

            <div class="result-card-content">
                {content}
            </div>
        </div>
        """,
        unsafe_allow_html=True,
    )


def metric_card(label, value):
    st.markdown(
        f"""
        <div class="metric-card">
            <div class="metric-label">
                {clean(label)}
            </div>

            <div class="metric-value">
                {clean(value)}
            </div>
        </div>
        """,
        unsafe_allow_html=True,
    )


def section_title(
    number,
    title,
    subtitle="",
):
    st.markdown(
        f"""
        <div class="section-heading">
            <span>{clean(number)}</span>

            <div>
                <h2>{clean(title)}</h2>
                <p>{clean(subtitle)}</p>
            </div>
        </div>
        """,
        unsafe_allow_html=True,
    )


def text_download(
    label,
    text,
    filename,
):
    st.download_button(
        label,
        data=str(text or ""),
        file_name=filename,
        mime="text/plain",
        use_container_width=True,
    )


# =========================================================
# GENRE OPTIONS
# =========================================================

GENRE_OPTIONS = [
    "Sci-Fi",
    "Space Opera",
    "Cosmic Horror",
    "Psychological Thriller",
    "Cyberpunk",
    "Fantasy",
    "Dark Fantasy",
    "Romance",
    "Action",
    "Mystery",
    "Crime",
    "Drama",
    "Comedy",
    "Horror",
    "Adventure",
    "Historical",
    "Custom Genre",
]


# =========================================================
# NAVIGATION HEADER
# =========================================================

NAV_ITEMS = [
    ("🏠", "Home"),
    ("✨", "Logline Forge"),
    ("🎭", "Character Vault"),
    ("💰", "Budget Desk"),
    ("📜", "Screenplay Lab"),
    ("🚀", "Master Blueprint"),
]


st.markdown(
    """
    <div class="top-brand">
        <div class="brand-mark">✦</div>

        <div>
            <div class="brand-name">
                AGENTIC AI CINEMA STUDIO
            </div>

            <div class="brand-subtitle">
                From one idea to a complete cinematic blueprint
            </div>
        </div>
    </div>
    """,
    unsafe_allow_html=True,
)


nav_cols = st.columns(
    len(NAV_ITEMS)
)

for col, (icon, page_name) in zip(
    nav_cols,
    NAV_ITEMS,
):
    with col:
        if st.button(
            f"{icon}  {page_name}",
            key=f"nav_{page_name}",
            use_container_width=True,
        ):
            go(page_name)


st.markdown(
    '<div class="nav-line"></div>',
    unsafe_allow_html=True,
)


# =========================================================
# HOME
# =========================================================

if st.session_state.page == "Home":

    st.markdown(
        """
        <div class="hero">
            <div class="hero-kicker">
                AI-POWERED STORY DEVELOPMENT
            </div>

            <h1>
                Turn an idea into a movie.
            </h1>

            <p>
                Build your logline, complete cast, production budget,
                screenplay, dialogue and final master blueprint —
                all from one story idea.
            </p>
        </div>
        """,
        unsafe_allow_html=True,
    )

    if not API_KEY:
        st.warning(
            "Gemini is not connected yet. "
            "Add GEMINI_API_KEY to .env to enable AI generation."
        )

    st.markdown(
        """
        <div class="workflow-strip">
            <span>IDEA</span><b>→</b>
            <span>LOGLINE</span><b>→</b>
            <span>CHARACTERS</span><b>→</b>
            <span>BUDGET</span><b>→</b>
            <span>SCREENPLAY</span><b>→</b>
            <span>MASTER BLUEPRINT</span>
        </div>
        """,
        unsafe_allow_html=True,
    )

    cards = [
        (
            "Logline Forge",
            "✨",
            IMG_LOGLINE,
            "Create three different logline directions so the user can choose the strongest one.",
            "Logline Forge",
        ),
        (
            "Character Vault",
            "🎭",
            IMG_CHARACTER,
            "Turn the selected logline into a complete ensemble. Regenerate any character independently.",
            "Character Vault",
        ),
        (
            "Budget Desk",
            "💰",
            IMG_BUDGET,
            "Choose the production scale with a slider and receive a detailed budget breakdown.",
            "Budget Desk",
        ),
        (
            "Screenplay Lab",
            "📜",
            IMG_SCREENPLAY,
            "Transform the selected logline into scenes, action, character cues and complete dialogue.",
            "Screenplay Lab",
        ),
    ]

    for row in range(2):

        c1, c2 = st.columns(2)

        for col, item in zip(
            (c1, c2),
            cards[row * 2 : row * 2 + 2],
        ):

            (
                title,
                icon,
                image,
                description,
                target,
            ) = item

            with col:

                st.markdown(
                    '<div class="feature-card">',
                    unsafe_allow_html=True,
                )

                show_image(
                    image,
                    title,
                )

                st.markdown(
                    f"""
                    <div class="feature-icon">
                        {icon}
                    </div>

                    <h3>
                        {clean(title)}
                    </h3>

                    <p>
                        {clean(description)}
                    </p>
                    """,
                    unsafe_allow_html=True,
                )

                if st.button(
                    f"Open {title} →",
                    key=f"home_{target}",
                    use_container_width=True,
                ):
                    go(target)

                st.markdown(
                    "</div>",
                    unsafe_allow_html=True,
                )

    st.markdown(
        """
        <div class="master-banner">
            <div>
                <span class="master-label">
                    FINAL STAGE
                </span>

                <h2>
                    🚀 Master Blueprint
                </h2>

                <p>
                    Combine the story, every character,
                    every scene, every dialogue and the
                    production plan into one downloadable
                    cinematic document.
                </p>
            </div>
        </div>
        """,
        unsafe_allow_html=True,
    )

    if st.button(
        "Open Master Blueprint →",
        key="home_master",
        use_container_width=True,
    ):
        go("Master Blueprint")


# =========================================================
# LOGLINE FORGE
# =========================================================

elif st.session_state.page == "Logline Forge":

    section_title(
        "01",
        "Logline Recommendation Engine",
        "Generate three distinct directions and let the user decide which story should move forward.",
    )

    c1, c2 = st.columns(2)

    with c1:

        selected_genre = st.selectbox(
            "Genre",
            GENRE_OPTIONS,
            index=(
                GENRE_OPTIONS.index(
                    st.session_state.project_genre
                )
                if st.session_state.project_genre
                in GENRE_OPTIONS
                else 0
            ),
            key="logline_genre",
        )

        if selected_genre == "Custom Genre":

            selected_genre = st.text_input(
                "Enter your genre",
                placeholder=(
                    "Example: Mythological cyber-noir romance"
                ),
                key="logline_custom_genre",
            )

    with c2:

        tone = st.select_slider(
            "Tone",
            options=[
                "Light",
                "Hopeful",
                "Cinematic",
                "Dark",
                "Intense",
                "Disturbing",
            ],
            value="Cinematic",
            key="logline_tone",
        )

    target = st.selectbox(
        "Target Audience",
        [
            "Mainstream theatrical audience",
            "Streaming audience",
            "Festival / art-house audience",
            "Young adult audience",
            "Family audience",
            "Adult audience",
        ],
        key="logline_audience",
    )

    story_seed = st.text_area(
        "Story idea / premise",
        placeholder=(
            "Describe your movie idea in your own words..."
        ),
        height=120,
        key="logline_story_seed",
    )

    if st.button(
        "✨ Generate 3 Logline Recommendations",
        use_container_width=True,
    ):

        if not selected_genre.strip():

            st.warning(
                "Please enter a genre."
            )

        elif not story_seed.strip():

            st.warning(
                "Please enter a story idea."
            )

        else:

            prompt = f"""
You are a professional film story development agent.

Create exactly 3 different logline recommendations
from the user's idea.

GENRE:
{selected_genre}

TONE:
{tone}

TARGET AUDIENCE:
{target}

USER STORY IDEA:
{story_seed}

The three options must be meaningfully different:

1. Commercial / high-concept hook
2. Character-driven / emotional hook
3. Bold / unconventional hook

Each must be a polished one- or two-sentence movie logline.

Do not write a screenplay.

Return ONLY valid JSON in this exact structure:

{{
  "options": [
    {{
      "title": "Option A",
      "logline": "...",
      "theme": "...",
      "hook": "...",
      "audience_reason": "..."
    }},
    {{
      "title": "Option B",
      "logline": "...",
      "theme": "...",
      "hook": "...",
      "audience_reason": "..."
    }},
    {{
      "title": "Option C",
      "logline": "...",
      "theme": "...",
      "hook": "...",
      "audience_reason": "..."
    }}
  ]
}}
"""

            result = ai_json(
                prompt,
                temperature=0.8,
            )

            if result:

                options = safe_list(
                    result.get("options")
                )

                # Keep only dictionary options.
                options = [
                    option
                    for option in options
                    if isinstance(option, dict)
                ]

                if not options:

                    st.error(
                        "Gemini returned no valid logline options."
                    )

                else:

                    st.session_state.logline_options = options

                    st.session_state.selected_logline = ""

                    st.session_state.project_genre = (
                        selected_genre
                    )

                    st.session_state.project_tone = tone

                    st.rerun()

    # -----------------------------------------------------
    # GENERATED OPTIONS
    # -----------------------------------------------------

    if st.session_state.logline_options:

        st.markdown("---")

        st.markdown(
            "### Choose the story direction"
        )

        for index, option in enumerate(
            st.session_state.logline_options
        ):

            title = as_text(
                option.get("title"),
                f"Option {index + 1}",
            )

            logline = as_text(
                option.get("logline")
            )

            theme = as_text(
                option.get("theme")
            )

            hook = as_text(
                option.get("hook")
            )

            audience = as_text(
                option.get("audience_reason")
            )

            st.markdown(
                f"""
                <div class="logline-option">

                    <div class="option-number">
                        {index + 1:02d}
                    </div>

                    <div class="option-body">

                        <div class="option-title">
                            {clean(title)}
                        </div>

                        <div class="option-logline">
                            “{clean(logline)}”
                        </div>

                        <div class="option-meta">
                            <b>Theme:</b>
                            {clean(theme)}

                            &nbsp;&nbsp;•&nbsp;&nbsp;

                            <b>Hook:</b>
                            {clean(hook)}
                        </div>

                        <div class="option-audience">
                            {clean(audience)}
                        </div>

                    </div>

                </div>
                """,
                unsafe_allow_html=True,
            )

            if st.button(
                f"✓ Choose {title}",
                key=f"choose_logline_{index}",
                use_container_width=True,
            ):

                st.session_state.selected_logline = (
                    logline
                )

                st.success(
                    f"{title} selected. "
                    "This is now the active story logline."
                )

    # -----------------------------------------------------
    # SELECTED LOGLINE
    # -----------------------------------------------------

    if st.session_state.selected_logline:

        card(
            "SELECTED LOGLINE",
            (
                f"<p>"
                f"{clean(st.session_state.selected_logline)}"
                f"</p>"
            ),
            "purple",
        )

        st.markdown(
            "### Continue the project"
        )

        c1, c2, c3 = st.columns(3)

        with c1:
            if st.button(
                "🎭 Build Characters",
                use_container_width=True,
            ):
                go("Character Vault")

        with c2:
            if st.button(
                "💰 Build Budget",
                use_container_width=True,
            ):
                go("Budget Desk")

        with c3:
            if st.button(
                "📜 Write Screenplay",
                use_container_width=True,
            ):
                go("Screenplay Lab")


# =========================================================
# CHARACTER VAULT
# =========================================================

elif st.session_state.page == "Character Vault":

    section_title(
        "02",
        "Character Vault",
        "Build the complete ensemble from the selected logline. Every character can be regenerated independently.",
    )

    current_logline = (
        st.session_state.selected_logline
    )

    if not current_logline:

        current_logline = st.text_area(
            "Paste your final logline",
            placeholder=(
                "Enter the approved logline here..."
            ),
            height=120,
            key="character_manual_logline",
        )

        if st.button(
            "Use this logline",
            use_container_width=True,
        ):

            if current_logline.strip():

                st.session_state.selected_logline = (
                    current_logline.strip()
                )

                st.rerun()

            else:

                st.warning(
                    "Please enter a logline."
                )

    else:

        card(
            "ACTIVE LOGLINE",
            f"<p>{clean(current_logline)}</p>",
            "cyan",
        )

    selected_genre = st.selectbox(
        "Genre",
        GENRE_OPTIONS,
        index=(
            GENRE_OPTIONS.index(
                st.session_state.project_genre
            )
            if st.session_state.project_genre
            in GENRE_OPTIONS
            else 0
        ),
        key="character_genre",
    )

    if selected_genre == "Custom Genre":

        selected_genre = st.text_input(
            "Enter custom genre",
            key="character_custom_genre",
        )

    cast_size = st.slider(
        "Number of main characters",
        3,
        12,
        6,
        key="character_cast_size",
    )

    if st.button(
        "🎭 Generate Complete Character Ensemble",
        use_container_width=True,
    ):

        if not current_logline.strip():

            st.warning(
                "Please provide a logline first."
            )

        elif not selected_genre.strip():

            st.warning(
                "Please provide a genre."
            )

        else:

            prompt = f"""
You are a professional character-development agent.

Create exactly {cast_size} important characters
for this movie.

GENRE:
{selected_genre}

LOGLINE:
{current_logline}

Include the protagonist, antagonist, co-leads,
supporting characters and any important role needed
by the story.

Do not create random filler characters.

Return ONLY valid JSON:

{{
  "characters": [
    {{
      "id": 1,
      "name": "...",
      "role": "...",
      "age": "...",
      "occupation": "...",
      "personality": "...",
      "goal": "...",
      "internal_need": "...",
      "flaw": "...",
      "backstory": "...",
      "character_arc": "...",
      "relationships": ["..."],
      "dialogue_voice": "...",
      "visual_identity": "..."
    }}
  ]
}}

Important:
- Return exactly {cast_size} characters.
- Every field must contain useful content.
- relationships must be an array of strings.
"""

            result = ai_json(
                prompt,
                temperature=0.8,
            )

            if result:

                characters = safe_list(
                    result.get("characters")
                )

                characters = [
                    character
                    for character in characters
                    if isinstance(character, dict)
                ]

                if not characters:

                    st.error(
                        "Gemini returned no valid characters."
                    )

                else:

                    st.session_state.characters = (
                        characters
                    )

                    st.session_state.project_genre = (
                        selected_genre
                    )

                    st.rerun()

    # -----------------------------------------------------
    # CHARACTER RESULTS
    # -----------------------------------------------------

    if st.session_state.characters:

        st.markdown("---")

        st.markdown(
            "### 🎭 Complete Ensemble"
        )

        for index, character in enumerate(
            st.session_state.characters
        ):

            character = safe_dict(
                character
            )

            name = as_text(
                character.get("name"),
                f"Character {index + 1}",
            )

            relationships = safe_list(
                character.get("relationships")
            )

            relationships_text = ", ".join(
                str(x)
                for x in relationships
            )

            with st.container():

                st.markdown(
                    f"""
                    <div class="character-card">

                        <div class="character-top">

                            <div>

                                <div class="character-role">
                                    {clean(
                                        character.get(
                                            "role",
                                            "CHARACTER"
                                        )
                                    )}
                                </div>

                                <h3>
                                    {clean(name)}
                                </h3>

                            </div>

                            <div class="character-number">
                                #{index + 1}
                            </div>

                        </div>

                        <div class="character-grid">

                            <div>
                                <b>Age</b>
                                <span>
                                    {clean(
                                        character.get("age")
                                    )}
                                </span>
                            </div>

                            <div>
                                <b>Occupation</b>
                                <span>
                                    {clean(
                                        character.get(
                                            "occupation"
                                        )
                                    )}
                                </span>
                            </div>

                            <div>
                                <b>Goal</b>
                                <span>
                                    {clean(
                                        character.get("goal")
                                    )}
                                </span>
                            </div>

                            <div>
                                <b>Need</b>
                                <span>
                                    {clean(
                                        character.get(
                                            "internal_need"
                                        )
                                    )}
                                </span>
                            </div>

                            <div>
                                <b>Flaw</b>
                                <span>
                                    {clean(
                                        character.get("flaw")
                                    )}
                                </span>
                            </div>

                            <div>
                                <b>Voice</b>
                                <span>
                                    {clean(
                                        character.get(
                                            "dialogue_voice"
                                        )
                                    )}
                                </span>
                            </div>

                        </div>

                        <div class="character-section">
                            <b>Backstory</b>
                            <p>
                                {clean(
                                    character.get(
                                        "backstory"
                                    )
                                )}
                            </p>
                        </div>

                        <div class="character-section">
                            <b>Character Arc</b>
                            <p>
                                {clean(
                                    character.get(
                                        "character_arc"
                                    )
                                )}
                            </p>
                        </div>

                        <div class="character-section">
                            <b>Relationships</b>
                            <p>
                                {clean(
                                    relationships_text
                                )}
                            </p>
                        </div>

                        <div class="character-section">
                            <b>Visual Identity</b>
                            <p>
                                {clean(
                                    character.get(
                                        "visual_identity"
                                    )
                                )}
                            </p>
                        </div>

                    </div>
                    """,
                    unsafe_allow_html=True,
                )

                if st.button(
                    f"🔄 Regenerate {name}",
                    key=f"regen_character_{index}",
                    use_container_width=True,
                ):

                    other_names = []

                    for i, c in enumerate(
                        st.session_state.characters
                    ):

                        if i == index:
                            continue

                        if isinstance(c, dict):
                            other_names.append(
                                as_text(
                                    c.get("name")
                                )
                            )

                    prompt = f"""
You are regenerating ONE character in an existing
film ensemble.

LOGLINE:
{current_logline}

GENRE:
{selected_genre}

OTHER CHARACTER NAMES:
{", ".join(other_names)}

CURRENT CHARACTER:
{json.dumps(
    character,
    ensure_ascii=False
)}

Create a better replacement character that fits
the story.

The replacement may have a completely new name.

Return ONLY valid JSON:

{{
  "id": {index + 1},
  "name": "...",
  "role": "...",
  "age": "...",
  "occupation": "...",
  "personality": "...",
  "goal": "...",
  "internal_need": "...",
  "flaw": "...",
  "backstory": "...",
  "character_arc": "...",
  "relationships": ["..."],
  "dialogue_voice": "...",
  "visual_identity": "..."
}}
"""

                    new_character = ai_json(
                        prompt,
                        temperature=0.85,
                    )

                    if new_character:

                        st.session_state.characters[
                            index
                        ] = new_character

                        st.rerun()

                st.markdown(
                    "<div class='card-gap'></div>",
                    unsafe_allow_html=True,
                )

        if st.button(
            "🔄 Regenerate Entire Ensemble",
            use_container_width=True,
        ):

            st.session_state.characters = []

            st.rerun()

        # -------------------------------------------------
        # DOWNLOAD CHARACTER VAULT
        # -------------------------------------------------

        character_text = (
            "CHARACTER VAULT\n\n"
        )

        for i, character in enumerate(
            st.session_state.characters,
            1,
        ):

            character = safe_dict(
                character
            )

            relationships = safe_list(
                character.get("relationships")
            )

            character_text += (
                f"{i}. "
                f"{as_text(character.get('name'))}\n"
                f"Role: "
                f"{as_text(character.get('role'))}\n"
                f"Age: "
                f"{as_text(character.get('age'))}\n"
                f"Occupation: "
                f"{as_text(character.get('occupation'))}\n"
                f"Personality: "
                f"{as_text(character.get('personality'))}\n"
                f"Goal: "
                f"{as_text(character.get('goal'))}\n"
                f"Internal Need: "
                f"{as_text(character.get('internal_need'))}\n"
                f"Flaw: "
                f"{as_text(character.get('flaw'))}\n"
                f"Backstory: "
                f"{as_text(character.get('backstory'))}\n"
                f"Character Arc: "
                f"{as_text(character.get('character_arc'))}\n"
                f"Relationships: "
                f"{', '.join(str(x) for x in relationships)}\n"
                f"Dialogue Voice: "
                f"{as_text(character.get('dialogue_voice'))}\n"
                f"Visual Identity: "
                f"{as_text(character.get('visual_identity'))}\n\n"
            )

        text_download(
            "⬇️ Download Character Vault",
            character_text,
            "character_vault.txt",
        )


# =========================================================
# BUDGET DESK
# =========================================================

elif st.session_state.page == "Budget Desk":

    section_title(
        "03",
        "Production Budget Desk",
        "Enter the approved logline and choose exactly how large you want the production to be.",
    )

    budget_logline = (
        st.session_state.selected_logline
    )

    if not budget_logline:

        budget_logline = st.text_area(
            "Movie logline",
            placeholder=(
                "Paste the final logline..."
            ),
            height=120,
            key="budget_manual_logline",
        )

    else:

        budget_logline = st.text_area(
            "Movie logline",
            value=budget_logline,
            height=120,
            key="budget_logline",
        )

    selected_genre = st.selectbox(
        "Genre",
        GENRE_OPTIONS,
        index=(
            GENRE_OPTIONS.index(
                st.session_state.project_genre
            )
            if st.session_state.project_genre
            in GENRE_OPTIONS
            else 0
        ),
        key="budget_genre",
    )

    if selected_genre == "Custom Genre":

        selected_genre = st.text_input(
            "Enter custom genre",
            key="budget_custom_genre",
        )

    budget_millions = st.slider(
        "Production Budget",
        min_value=1,
        max_value=300,
        value=50,
        step=1,
        format="$%dM",
        key="budget_amount",
    )

    budget_style = st.radio(
        "Budget strategy",
        [
            "Lean / cost-efficient",
            "Balanced",
            "Premium / quality-first",
        ],
        horizontal=True,
        key="budget_strategy",
    )

    st.markdown(
        f"""
        <div class="budget-preview">

            <span>
                SELECTED PRODUCTION SCALE
            </span>

            <strong>
                ${budget_millions} Million
            </strong>

            <p>
                {clean(budget_style)}
            </p>

        </div>
        """,
        unsafe_allow_html=True,
    )

    if st.button(
        "💰 Generate Detailed Budget",
        use_container_width=True,
    ):

        if not budget_logline.strip():

            st.warning(
                "Please provide a logline."
            )

        elif not selected_genre.strip():

            st.warning(
                "Please provide a genre."
            )

        else:

            prompt = f"""
You are a professional film production budgeting agent.

Prepare a realistic planning-level film budget.

LOGLINE:
{budget_logline}

GENRE:
{selected_genre}

TOTAL TARGET BUDGET:
${budget_millions} million USD

BUDGET STRATEGY:
{budget_style}

Create department-level allocations.

The percentages MUST add to exactly 100%.

Adapt the allocation to the genre and story instead
of using a generic fixed template.

Return ONLY valid JSON:

{{
  "total_budget": "{budget_millions} million USD",
  "summary": "...",
  "departments": [
    {{
      "department": "...",
      "percentage": 0,
      "amount": "...",
      "details": [
        "...",
        "...",
        "..."
      ]
    }}
  ],
  "risk_analysis": [
    "...",
    "...",
    "..."
  ],
  "cost_saving_options": [
    "...",
    "...",
    "..."
  ]
}}
"""

            result = ai_json(
                prompt,
                temperature=0.5,
            )

            if result:

                departments = safe_list(
                    result.get("departments")
                )

                departments = [
                    dept
                    for dept in departments
                    if isinstance(dept, dict)
                ]

                result["departments"] = departments

                result["risk_analysis"] = (
                    safe_list(
                        result.get(
                            "risk_analysis"
                        )
                    )
                )

                result["cost_saving_options"] = (
                    safe_list(
                        result.get(
                            "cost_saving_options"
                        )
                    )
                )

                st.session_state.budget_result = (
                    result
                )

                st.session_state.project_genre = (
                    selected_genre
                )

                st.rerun()

    # -----------------------------------------------------
    # BUDGET RESULTS
    # -----------------------------------------------------

    if st.session_state.budget_result:

        result = safe_dict(
            st.session_state.budget_result
        )

        st.markdown("---")

        st.markdown(
            "### 💰 Budget Overview"
        )

        c1, c2, c3 = st.columns(3)

        with c1:
            metric_card(
                "TOTAL BUDGET",
                as_text(
                    result.get(
                        "total_budget",
                        f"${budget_millions}M",
                    )
                ),
            )

        with c2:
            metric_card(
                "GENRE",
                selected_genre,
            )

        with c3:
            metric_card(
                "STRATEGY",
                budget_style,
            )

        card(
            "Production Summary",
            (
                f"<p>"
                f"{clean(result.get('summary'))}"
                f"</p>"
            ),
            "cyan",
        )

        st.markdown(
            "### Department Allocation"
        )

        for dept in safe_list(
            result.get("departments")
        ):

            if not isinstance(dept, dict):
                continue

            details = "".join(
                f"<li>{clean(x)}</li>"
                for x in safe_list(
                    dept.get("details")
                )
            )

            st.markdown(
                f"""
                <div class="budget-row">

                    <div>

                        <h3>
                            {clean(
                                dept.get(
                                    "department"
                                )
                            )}
                        </h3>

                        <p>
                            {details}
                        </p>

                    </div>

                    <div class="budget-amount">

                        <strong>
                            {clean(
                                dept.get("amount")
                            )}
                        </strong>

                        <span>
                            {clean(
                                dept.get(
                                    "percentage"
                                )
                            )}%
                        </span>

                    </div>

                </div>
                """,
                unsafe_allow_html=True,
            )

        risk = "".join(
            f"<li>{clean(x)}</li>"
            for x in safe_list(
                result.get("risk_analysis")
            )
        )

        savings = "".join(
            f"<li>{clean(x)}</li>"
            for x in safe_list(
                result.get(
                    "cost_saving_options"
                )
            )
        )

        c1, c2 = st.columns(2)

        with c1:
            card(
                "Risk Analysis",
                f"<ul>{risk}</ul>",
                "red",
            )

        with c2:
            card(
                "Cost-Saving Options",
                f"<ul>{savings}</ul>",
                "purple",
            )

        # -------------------------------------------------
        # DOWNLOAD BUDGET
        # -------------------------------------------------

        budget_text = (
            "PRODUCTION BUDGET\n\n"
            f"Total: "
            f"{as_text(result.get('total_budget'))}\n"
            f"Genre: {selected_genre}\n"
            f"Strategy: {budget_style}\n\n"
            f"Summary:\n"
            f"{as_text(result.get('summary'))}\n\n"
            "DEPARTMENTS\n"
        )

        for dept in safe_list(
            result.get("departments")
        ):

            if not isinstance(dept, dict):
                continue

            budget_text += (
                f"\n"
                f"{as_text(dept.get('department'))}"
                f" — "
                f"{as_text(dept.get('percentage'))}%"
                f" — "
                f"{as_text(dept.get('amount'))}\n"
            )

            for detail in safe_list(
                dept.get("details")
            ):

                budget_text += (
                    f"  - {detail}\n"
                )

        budget_text += (
            "\nRISK ANALYSIS\n"
        )

        for item in safe_list(
            result.get("risk_analysis")
        ):

            budget_text += (
                f"- {item}\n"
            )

        budget_text += (
            "\nCOST-SAVING OPTIONS\n"
        )

        for item in safe_list(
            result.get(
                "cost_saving_options"
            )
        ):

            budget_text += (
                f"- {item}\n"
            )

        text_download(
            "⬇️ Download Production Budget",
            budget_text,
            "production_budget.txt",
        )


# =========================================================
# SCREENPLAY LAB
# =========================================================

elif st.session_state.page == "Screenplay Lab":

    section_title(
        "04",
        "Screenplay Lab",
        "Generate a complete scene-by-scene screenplay from the approved logline.",
    )

    screenplay_logline = (
        st.session_state.selected_logline
    )

    if not screenplay_logline:

        screenplay_logline = st.text_area(
            "Movie logline",
            placeholder=(
                "Paste the final logline..."
            ),
            height=120,
            key="screenplay_manual_logline",
        )

    else:

        screenplay_logline = st.text_area(
            "Movie logline",
            value=screenplay_logline,
            height=120,
            key="screenplay_logline",
        )

    selected_genre = st.selectbox(
        "Genre",
        GENRE_OPTIONS,
        index=(
            GENRE_OPTIONS.index(
                st.session_state.project_genre
            )
            if st.session_state.project_genre
            in GENRE_OPTIONS
            else 0
        ),
        key="screenplay_genre",
    )

    if selected_genre == "Custom Genre":

        selected_genre = st.text_input(
            "Enter custom genre",
            key="screenplay_custom_genre",
        )

    c1, c2 = st.columns(2)

    with c1:

        scene_count = st.slider(
            "Number of scenes",
            5,
            20,
            10,
            key="screenplay_scene_count",
        )

    with c2:

        screenplay_length = st.select_slider(
            "Scene detail",
            options=[
                "Compact",
                "Detailed",
                "Very Detailed",
            ],
            value="Detailed",
            key="screenplay_detail",
        )

    if st.button(
        "📜 Generate Complete Screenplay",
        use_container_width=True,
    ):

        if not screenplay_logline.strip():

            st.warning(
                "Please provide a logline."
            )

        elif not selected_genre.strip():

            st.warning(
                "Please provide a genre."
            )

        else:

            prompt = f"""
You are a professional screenwriter.

Write a coherent screenplay based ONLY on this
story foundation.

LOGLINE:
{screenplay_logline}

GENRE:
{selected_genre}

NUMBER OF SCENES:
{scene_count}

DETAIL LEVEL:
{screenplay_length}

Create exactly {scene_count} scenes across three acts.

Every scene must contain:

- scene number
- INT./EXT. heading
- location
- time
- action/visual description
- characters present
- meaningful dialogue
- parentheticals only when useful
- scene purpose / progression

The story must have continuity.

Characters should behave consistently.

Do not skip scenes.

Do not summarize the dialogue.

Give actual dialogue for every important exchange.

Return ONLY valid JSON:

{{
  "title": "...",
  "genre": "...",
  "logline": "...",
  "acts": [
    {{
      "act": "ACT I",
      "purpose": "...",
      "scenes": [
        {{
          "scene_number": 1,
          "heading": "INT. ... - NIGHT",
          "location": "...",
          "time": "...",
          "characters": ["..."],
          "action": "...",
          "dialogue": [
            {{
              "character": "...",
              "parenthetical": "",
              "line": "..."
            }}
          ],
          "scene_purpose": "..."
        }}
      ]
    }}
  ]
}}
"""

            result = ai_json(
                prompt,
                temperature=0.85,
            )

            if result:

                acts = safe_list(
                    result.get("acts")
                )

                valid_acts = []

                for act in acts:

                    if not isinstance(act, dict):
                        continue

                    scenes = safe_list(
                        act.get("scenes")
                    )

                    valid_scenes = [
                        scene
                        for scene in scenes
                        if isinstance(scene, dict)
                    ]

                    act["scenes"] = valid_scenes

                    valid_acts.append(act)

                result["acts"] = valid_acts

                st.session_state.screenplay_result = (
                    result
                )

                st.session_state.project_genre = (
                    selected_genre
                )

                st.rerun()

    # -----------------------------------------------------
    # SCREENPLAY RESULTS
    # -----------------------------------------------------

    if st.session_state.screenplay_result:

        result = safe_dict(
            st.session_state.screenplay_result
        )

        st.markdown("---")

        st.markdown(
            f"""
            <div class="screenplay-title">

                <span>
                    {clean(
                        result.get(
                            "genre",
                            selected_genre,
                        )
                    )}
                </span>

                <h2>
                    {clean(
                        result.get(
                            "title",
                            "Untitled Screenplay",
                        )
                    )}
                </h2>

                <p>
                    {clean(
                        result.get(
                            "logline",
                            screenplay_logline,
                        )
                    )}
                </p>

            </div>
            """,
            unsafe_allow_html=True,
        )

        for act in safe_list(
            result.get("acts")
        ):

            if not isinstance(act, dict):
                continue

            st.markdown(
                f"""
                <div class="act-header">
                    {clean(
                        act.get("act")
                    )}

                    <span>
                        {clean(
                            act.get("purpose")
                        )}
                    </span>
                </div>
                """,
                unsafe_allow_html=True,
            )

            for scene in safe_list(
                act.get("scenes")
            ):

                if not isinstance(scene, dict):
                    continue

                st.markdown(
                    f"""
                    <div class="scene-block">

                        <div class="scene-heading">
                            {clean(
                                scene.get(
                                    "heading"
                                )
                            )}
                        </div>

                        <div class="scene-location">
                            {clean(
                                scene.get(
                                    "location"
                                )
                            )}
                            •
                            {clean(
                                scene.get(
                                    "time"
                                )
                            )}
                        </div>

                        <div class="action-text">
                            {clean(
                                scene.get(
                                    "action"
                                )
                            )}
                        </div>
                    """,
                    unsafe_allow_html=True,
                )

                for line in safe_list(
                    scene.get("dialogue")
                ):

                    if not isinstance(line, dict):
                        continue

                    parenthetical = as_text(
                        line.get(
                            "parenthetical"
                        )
                    )

                    if parenthetical:

                        p_html = (
                            f"""
                            <div class="parenthetical">
                                ({clean(parenthetical)})
                            </div>
                            """
                        )

                    else:

                        p_html = ""

                    st.markdown(
                        f"""
                        <div class="dialogue-block">

                            <div class="character-cue">
                                {clean(
                                    line.get(
                                        "character"
                                    )
                                )}
                            </div>

                            {p_html}

                            <div class="dialogue-text">
                                {clean(
                                    line.get(
                                        "line"
                                    )
                                )}
                            </div>

                        </div>
                        """,
                        unsafe_allow_html=True,
                    )

                st.markdown(
                    f"""
                    <div class="scene-purpose">
                        <b>Scene purpose:</b>
                        {clean(
                            scene.get(
                                "scene_purpose"
                            )
                        )}
                    </div>

                    </div>
                    """,
                    unsafe_allow_html=True,
                )

        # -------------------------------------------------
        # DOWNLOAD SCREENPLAY
        # -------------------------------------------------

        screenplay_text = (
            f"{as_text(result.get('title'), 'Untitled')}\n"
            f"GENRE: "
            f"{as_text(result.get('genre'), selected_genre)}\n"
            f"LOGLINE: "
            f"{as_text(result.get('logline'), screenplay_logline)}\n\n"
        )

        for act in safe_list(
            result.get("acts")
        ):

            if not isinstance(act, dict):
                continue

            screenplay_text += (
                f"\n\n"
                f"{as_text(act.get('act'))}\n"
                f"{as_text(act.get('purpose'))}\n\n"
            )

            for scene in safe_list(
                act.get("scenes")
            ):

                if not isinstance(scene, dict):
                    continue

                screenplay_text += (
                    f"SCENE "
                    f"{as_text(scene.get('scene_number'))}"
                    f" — "
                    f"{as_text(scene.get('heading'))}\n"
                    f"{as_text(scene.get('location'))}"
                    f" • "
                    f"{as_text(scene.get('time'))}\n\n"
                    f"{as_text(scene.get('action'))}\n\n"
                )

                for line in safe_list(
                    scene.get("dialogue")
                ):

                    if not isinstance(line, dict):
                        continue

                    character = as_text(
                        line.get("character")
                    )

                    parenthetical = as_text(
                        line.get(
                            "parenthetical"
                        )
                    )

                    dialogue_line = as_text(
                        line.get("line")
                    )

                    screenplay_text += (
                        f"        {character}\n"
                    )

                    if parenthetical:

                        screenplay_text += (
                            f"        "
                            f"({parenthetical})\n"
                        )

                    screenplay_text += (
                        f"        "
                        f"{dialogue_line}\n\n"
                    )

                screenplay_text += (
                    f"Scene purpose: "
                    f"{as_text(scene.get('scene_purpose'))}"
                    f"\n\n"
                )

        text_download(
            "⬇️ Download Complete Screenplay",
            screenplay_text,
            "complete_screenplay.txt",
        )


# =========================================================
# MASTER BLUEPRINT
# =========================================================

elif st.session_state.page == "Master Blueprint":

    section_title(
        "05",
        "Master Blueprint",
        "The final production document: characters, screenplay, dialogue, budget and story structure in one place.",
    )

    blueprint_logline = (
        st.session_state.selected_logline
    )

    if not blueprint_logline:

        blueprint_logline = st.text_area(
            "Final movie logline",
            placeholder=(
                "Paste the approved final logline..."
            ),
            height=130,
            key="master_manual_logline",
        )

    else:

        blueprint_logline = st.text_area(
            "Final movie logline",
            value=blueprint_logline,
            height=130,
            key="master_logline",
        )

    selected_genre = st.selectbox(
        "Genre",
        GENRE_OPTIONS,
        index=(
            GENRE_OPTIONS.index(
                st.session_state.project_genre
            )
            if st.session_state.project_genre
            in GENRE_OPTIONS
            else 0
        ),
        key="master_genre",
    )

    if selected_genre == "Custom Genre":

        selected_genre = st.text_input(
            "Enter custom genre",
            key="master_custom_genre",
        )

    c1, c2, c3 = st.columns(3)

    with c1:

        cast_size = st.slider(
            "Characters",
            3,
            12,
            7,
            key="master_cast_size",
        )

    with c2:

        scene_count = st.slider(
            "Scenes",
            6,
            24,
            12,
            key="master_scene_count",
        )

    with c3:

        budget_millions = st.slider(
            "Budget ($M)",
            1,
            300,
            50,
            key="master_budget",
        )

    tone = st.select_slider(
        "Overall cinematic tone",
        options=[
            "Warm",
            "Hopeful",
            "Cinematic",
            "Dark",
            "Intense",
            "Bleak",
        ],
        value="Cinematic",
        key="master_tone",
    )

    st.info(
        "The Master Blueprint is intentionally comprehensive. "
        "A large blueprint can use a significant amount of Gemini output quota."
    )

    if st.button(
        "🚀 Generate Complete Master Blueprint",
        use_container_width=True,
    ):

        if not blueprint_logline.strip():

            st.warning(
                "Please provide the final logline."
            )

        elif not selected_genre.strip():

            st.warning(
                "Please provide a genre."
            )

        else:

            prompt = f"""
You are the MASTER AI FILM DEVELOPMENT AGENT.

Build a complete movie blueprint from this approved
logline.

LOGLINE:
{blueprint_logline}

GENRE:
{selected_genre}

TONE:
{tone}

MAIN CHARACTERS:
{cast_size}

SCENES:
{scene_count}

TARGET PRODUCTION BUDGET:
${budget_millions} million USD

The result must be a complete production blueprint,
not a short summary.

Include:

1. Movie title
2. One-sentence logline
3. Theme
4. Tone
5. World / setting
6. Full cast with every important character
7. Character goals, flaws, arcs and relationships
8. Three-act story structure
9. Every scene in sequence
10. Action and visual direction for every scene
11. Every important dialogue exchange
12. Production budget with department allocation
13. Key production risks
14. Final ending / resolution

Do not say:
- etc.
- and so on
- dialogue continues
- more scenes

Do not leave placeholders.

Actually write the content.

Return ONLY valid JSON:

{{
  "title": "...",
  "logline": "...",
  "theme": "...",
  "tone": "...",
  "world": "...",

  "characters": [
    {{
      "name": "...",
      "role": "...",
      "age": "...",
      "personality": "...",
      "goal": "...",
      "need": "...",
      "flaw": "...",
      "backstory": "...",
      "arc": "...",
      "relationships": "...",
      "dialogue_voice": "..."
    }}
  ],

  "acts": [
    {{
      "act": "ACT I",
      "purpose": "...",
      "scenes": [
        {{
          "scene_number": 1,
          "heading": "INT./EXT. ... - DAY/NIGHT",
          "action": "...",
          "characters": ["..."],
          "dialogue": [
            {{
              "character": "...",
              "parenthetical": "",
              "line": "..."
            }}
          ],
          "purpose": "..."
        }}
      ]
    }}
  ],

  "budget": {{
    "total": "${budget_millions} million USD",
    "departments": [
      {{
        "department": "...",
        "percentage": 0,
        "amount": "...",
        "details": "..."
      }}
    ],
    "risks": [
      "...",
      "...",
      "..."
    ]
  }},

  "ending": "..."
}}
"""

            result = ai_json(
                prompt,
                temperature=0.8,
            )

            if result:

                # -----------------------------------------
                # NORMALIZE MASTER DATA
                # -----------------------------------------

                characters = safe_list(
                    result.get("characters")
                )

                result["characters"] = [
                    character
                    for character in characters
                    if isinstance(character, dict)
                ]

                acts = safe_list(
                    result.get("acts")
                )

                valid_acts = []

                for act in acts:

                    if not isinstance(act, dict):
                        continue

                    scenes = safe_list(
                        act.get("scenes")
                    )

                    act["scenes"] = [
                        scene
                        for scene in scenes
                        if isinstance(scene, dict)
                    ]

                    valid_acts.append(act)

                result["acts"] = valid_acts

                budget = safe_dict(
                    result.get("budget")
                )

                departments = safe_list(
                    budget.get("departments")
                )

                budget["departments"] = [
                    dept
                    for dept in departments
                    if isinstance(dept, dict)
                ]

                budget["risks"] = safe_list(
                    budget.get("risks")
                )

                result["budget"] = budget

                st.session_state.blueprint_result = (
                    result
                )

                st.session_state.project_genre = (
                    selected_genre
                )

                st.session_state.project_tone = (
                    tone
                )

                st.rerun()

    # -----------------------------------------------------
    # BLUEPRINT RESULTS
    # -----------------------------------------------------

    if st.session_state.blueprint_result:

        result = safe_dict(
            st.session_state.blueprint_result
        )

        st.markdown("---")

        st.markdown(
            f"""
            <div class="blueprint-hero">

                <div class="blueprint-label">
                    MASTER PRODUCTION BLUEPRINT
                </div>

                <h1>
                    {clean(
                        result.get(
                            "title",
                            "Untitled Film",
                        )
                    )}
                </h1>

                <p>
                    {clean(
                        result.get(
                            "logline",
                            blueprint_logline,
                        )
                    )}
                </p>

            </div>
            """,
            unsafe_allow_html=True,
        )

        characters = safe_list(
            result.get("characters")
        )

        acts = safe_list(
            result.get("acts")
        )

        budget = safe_dict(
            result.get("budget")
        )

        total_scenes = 0

        for act in acts:

            if isinstance(act, dict):

                total_scenes += len(
                    [
                        scene
                        for scene in safe_list(
                            act.get("scenes")
                        )
                        if isinstance(scene, dict)
                    ]
                )

        c1, c2, c3, c4 = st.columns(4)

        with c1:

            metric_card(
                "GENRE",
                selected_genre,
            )

        with c2:

            metric_card(
                "CHARACTERS",
                len(characters),
            )

        with c3:

            metric_card(
                "SCENES",
                total_scenes,
            )

        with c4:

            metric_card(
                "BUDGET",
                as_text(
                    budget.get(
                        "total",
                        f"${budget_millions}M",
                    )
                ),
            )

        card(
            "Theme",
            (
                f"<p>"
                f"{clean(result.get('theme'))}"
                f"</p>"
            ),
            "purple",
        )

        card(
            "World & Setting",
            (
                f"<p>"
                f"{clean(result.get('world'))}"
                f"</p>"
            ),
            "cyan",
        )

        # -------------------------------------------------
        # CHARACTERS
        # -------------------------------------------------

        st.markdown(
            "## 🎭 Every Character"
        )

        for i, character in enumerate(
            characters,
            1,
        ):

            if not isinstance(
                character,
                dict,
            ):
                continue

            relationships = character.get(
                "relationships"
            )

            if isinstance(
                relationships,
                list,
            ):

                relationships = ", ".join(
                    str(x)
                    for x in relationships
                )

            st.markdown(
                f"""
                <div class="blueprint-character">

                    <div class="character-number">
                        #{i}
                    </div>

                    <div>

                        <div class="character-role">
                            {clean(
                                character.get(
                                    "role"
                                )
                            )}
                        </div>

                        <h3>
                            {clean(
                                character.get(
                                    "name"
                                )
                            )}
                        </h3>

                        <p>
                            <b>Age:</b>
                            {clean(
                                character.get(
                                    "age"
                                )
                            )}
                        </p>

                        <p>
                            <b>Personality:</b>
                            {clean(
                                character.get(
                                    "personality"
                                )
                            )}
                        </p>

                        <p>
                            <b>Goal:</b>
                            {clean(
                                character.get(
                                    "goal"
                                )
                            )}
                        </p>

                        <p>
                            <b>Need:</b>
                            {clean(
                                character.get(
                                    "need"
                                )
                            )}
                        </p>

                        <p>
                            <b>Flaw:</b>
                            {clean(
                                character.get(
                                    "flaw"
                                )
                            )}
                        </p>

                        <p>
                            <b>Backstory:</b>
                            {clean(
                                character.get(
                                    "backstory"
                                )
                            )}
                        </p>

                        <p>
                            <b>Arc:</b>
                            {clean(
                                character.get(
                                    "arc"
                                )
                            )}
                        </p>

                        <p>
                            <b>Relationships:</b>
                            {clean(
                                relationships
                            )}
                        </p>

                        <p>
                            <b>Dialogue Voice:</b>
                            {clean(
                                character.get(
                                    "dialogue_voice"
                                )
                            )}
                        </p>

                    </div>

                </div>
                """,
                unsafe_allow_html=True,
            )

        # -------------------------------------------------
        # SCREENPLAY
        # -------------------------------------------------

        st.markdown(
            "## 🎬 Complete Screenplay & Dialogue"
        )

        for act in acts:

            if not isinstance(
                act,
                dict,
            ):
                continue

            st.markdown(
                f"""
                <div class="act-header">

                    {clean(
                        act.get("act")
                    )}

                    <span>
                        {clean(
                            act.get("purpose")
                        )}
                    </span>

                </div>
                """,
                unsafe_allow_html=True,
            )

            for scene in safe_list(
                act.get("scenes")
            ):

                if not isinstance(
                    scene,
                    dict,
                ):
                    continue

                st.markdown(
                    f"""
                    <div class="scene-block">

                        <div class="scene-heading">
                            SCENE
                            {clean(
                                scene.get(
                                    "scene_number"
                                )
                            )}
                            —
                            {clean(
                                scene.get(
                                    "heading"
                                )
                            )}
                        </div>

                        <div class="action-text">
                            {clean(
                                scene.get(
                                    "action"
                                )
                            )}
                        </div>
                    """,
                    unsafe_allow_html=True,
                )

                for line in safe_list(
                    scene.get("dialogue")
                ):

                    if not isinstance(
                        line,
                        dict,
                    ):
                        continue

                    parenthetical = as_text(
                        line.get(
                            "parenthetical"
                        )
                    )

                    if parenthetical:

                        p_html = (
                            f"""
                            <div class="parenthetical">
                                ({clean(parenthetical)})
                            </div>
                            """
                        )

                    else:

                        p_html = ""

                    st.markdown(
                        f"""
                        <div class="dialogue-block">

                            <div class="character-cue">
                                {clean(
                                    line.get(
                                        "character"
                                    )
                                )}
                            </div>

                            {p_html}

                            <div class="dialogue-text">
                                {clean(
                                    line.get(
                                        "line"
                                    )
                                )}
                            </div>

                        </div>
                        """,
                        unsafe_allow_html=True,
                    )

                st.markdown(
                    f"""
                    <div class="scene-purpose">

                        <b>Scene purpose:</b>

                        {clean(
                            scene.get(
                                "purpose"
                            )
                        )}

                    </div>

                    </div>
                    """,
                    unsafe_allow_html=True,
                )

        # -------------------------------------------------
        # PRODUCTION BUDGET
        # -------------------------------------------------

        st.markdown(
            "## 💰 Production Budget"
        )

        for dept in safe_list(
            budget.get("departments")
        ):

            if not isinstance(
                dept,
                dict,
            ):
                continue

            st.markdown(
                f"""
                <div class="budget-row">

                    <div>

                        <h3>
                            {clean(
                                dept.get(
                                    "department"
                                )
                            )}
                        </h3>

                        <p>
                            {clean(
                                dept.get(
                                    "details"
                                )
                            )}
                        </p>

                    </div>

                    <div class="budget-amount">

                        <strong>
                            {clean(
                                dept.get(
                                    "amount"
                                )
                            )}
                        </strong>

                        <span>
                            {clean(
                                dept.get(
                                    "percentage"
                                )
                            )}%
                        </span>

                    </div>

                </div>
                """,
                unsafe_allow_html=True,
            )

        risks = "".join(
            f"<li>{clean(x)}</li>"
            for x in safe_list(
                budget.get("risks")
            )
        )

        card(
            "Production Risks",
            f"<ul>{risks}</ul>",
            "red",
        )

        card(
            "Final Ending",
            (
                f"<p>"
                f"{clean(result.get('ending'))}"
                f"</p>"
            ),
            "purple",
        )

        # -------------------------------------------------
        # DOWNLOAD MASTER BLUEPRINT
        # -------------------------------------------------

        master_text = (
            "============================================================\n"
            "MASTER PRODUCTION BLUEPRINT\n"
            "============================================================\n\n"
            f"TITLE: "
            f"{as_text(result.get('title'))}\n"
            f"GENRE: {selected_genre}\n"
            f"TONE: "
            f"{as_text(result.get('tone'))}\n\n"
            f"LOGLINE:\n"
            f"{as_text(result.get('logline'))}\n\n"
            f"THEME:\n"
            f"{as_text(result.get('theme'))}\n\n"
            f"WORLD / SETTING:\n"
            f"{as_text(result.get('world'))}\n\n"
            "============================================================\n"
            "CHARACTERS\n"
            "============================================================\n\n"
        )

        for i, character in enumerate(
            characters,
            1,
        ):

            if not isinstance(
                character,
                dict,
            ):
                continue

            relationships = character.get(
                "relationships"
            )

            if isinstance(
                relationships,
                list,
            ):

                relationships = ", ".join(
                    str(x)
                    for x in relationships
                )

            master_text += (
                f"{i}. "
                f"{as_text(character.get('name'))}\n"
                f"Role: "
                f"{as_text(character.get('role'))}\n"
                f"Age: "
                f"{as_text(character.get('age'))}\n"
                f"Personality: "
                f"{as_text(character.get('personality'))}\n"
                f"Goal: "
                f"{as_text(character.get('goal'))}\n"
                f"Need: "
                f"{as_text(character.get('need'))}\n"
                f"Flaw: "
                f"{as_text(character.get('flaw'))}\n"
                f"Backstory: "
                f"{as_text(character.get('backstory'))}\n"
                f"Arc: "
                f"{as_text(character.get('arc'))}\n"
                f"Relationships: "
                f"{as_text(relationships)}\n"
                f"Dialogue Voice: "
                f"{as_text(character.get('dialogue_voice'))}\n\n"
            )

        master_text += (
            "============================================================\n"
            "COMPLETE SCREENPLAY\n"
            "============================================================\n"
        )

        for act in acts:

            if not isinstance(
                act,
                dict,
            ):
                continue

            master_text += (
                f"\n\n"
                f"{as_text(act.get('act'))}\n"
                f"{as_text(act.get('purpose'))}\n\n"
            )

            for scene in safe_list(
                act.get("scenes")
            ):

                if not isinstance(
                    scene,
                    dict,
                ):
                    continue

                master_text += (
                    f"SCENE "
                    f"{as_text(scene.get('scene_number'))}"
                    f" — "
                    f"{as_text(scene.get('heading'))}\n\n"
                    f"{as_text(scene.get('action'))}\n\n"
                )

                for line in safe_list(
                    scene.get("dialogue")
                ):

                    if not isinstance(
                        line,
                        dict,
                    ):
                        continue

                    master_text += (
                        f"{as_text(line.get('character'))}\n"
                    )

                    parenthetical = as_text(
                        line.get(
                            "parenthetical"
                        )
                    )

                    if parenthetical:

                        master_text += (
                            f"({parenthetical})\n"
                        )

                    master_text += (
                        f"{as_text(line.get('line'))}\n\n"
                    )

                master_text += (
                    f"Scene Purpose: "
                    f"{as_text(scene.get('purpose'))}\n\n"
                )

        master_text += (
            "============================================================\n"
            "PRODUCTION BUDGET\n"
            "============================================================\n\n"
            f"TOTAL: "
            f"{as_text(budget.get('total'))}\n\n"
        )

        for dept in safe_list(
            budget.get("departments")
        ):

            if not isinstance(
                dept,
                dict,
            ):
                continue

            master_text += (
                f"{as_text(dept.get('department'))}"
                f" — "
                f"{as_text(dept.get('percentage'))}%"
                f" — "
                f"{as_text(dept.get('amount'))}\n"
                f"{as_text(dept.get('details'))}\n\n"
            )

        master_text += (
            "PRODUCTION RISKS\n"
        )

        for risk in safe_list(
            budget.get("risks")
        ):

            master_text += (
                f"- {risk}\n"
            )

        master_text += (
            "\n============================================================\n"
            "ENDING\n"
            "============================================================\n\n"
            f"{as_text(result.get('ending'))}\n"
        )

        text_download(
            "⬇️ DOWNLOAD COMPLETE MASTER BLUEPRINT",
            master_text,
            "master_production_blueprint.txt",
        )