import streamlit as st
import sys
from pathlib import Path
import os
import random

# ─── PATH SETUP ──────────────────────────────────────────────────────────────
sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

from dotenv import load_dotenv

# Chemin explicite vers le .env (racine du projet) — on ne laisse pas dotenv
# deviner l'emplacement, car cela dépend de la façon dont le script est lancé.
APP_DIR = Path(__file__).resolve().parent
PROJECT_ROOT = APP_DIR.parent
ENV_PATH = PROJECT_ROOT / ".env"
load_dotenv(dotenv_path=ENV_PATH)

from embeddings.embedder import Embedder
from retrieval.vector_store import VectorStore
from retrieval.reranker import Reranker
from generation.llm import GroqLLM
from pipeline.rag_pipeline import ClimateRAG

FAISS_INDEX_PATH = os.getenv("FAISS_INDEX_PATH", "data/index/climate_index")

# ─── PAGE CONFIG ─────────────────────────────────────────────────────────────
st.set_page_config(
    page_title="ClimateRAG — Assistant Scientifique Climat",
    page_icon="🌍",
    layout="wide",
    initial_sidebar_state="expanded",
)

# ─── DESIGN SYSTEM ────────────────────────────────────────────────────────────
# Palette grounded in climate science: a neutral "paper" background, a cool
# teal (science / past) and a warm burnt-orange (heat / urgency) accent, tied
# together by a signature gradient bar referencing the classic climate
# "warming stripes" — the same blue-to-red scale scientists use for anomalies.
st.markdown("""
<style>
@import url('https://fonts.googleapis.com/css2?family=Fraunces:opsz,wght@9..144,400;9..144,500;9..144,600;9..144,700&family=Inter:wght@400;500;600;700&family=IBM+Plex+Mono:wght@400;500&display=swap');

:root {
    --bg: #F6F7F4;
    --surface: #FFFFFF;
    --surface-muted: #EDF1EA;
    --border: #E1E6DD;
    --text: #1C2620;
    --text-muted: #64705F;
    --accent-cool: #175E68;
    --accent-cool-soft: rgba(23, 94, 104, 0.08);
    --accent-warm: #C2410C;
    --accent-warm-soft: rgba(194, 65, 12, 0.08);
    --success: #2F9E5C;
    --error: #B3261E;
    --stripes: linear-gradient(90deg, #08306B 0%, #2166AC 16%, #92C5DE 32%, #F2F1EA 50%, #F4A582 68%, #B2182B 84%, #67001F 100%);
    --radius-lg: 18px;
    --radius-md: 12px;
    --radius-sm: 8px;
    --shadow-card: 0 1px 2px rgba(28, 38, 32, 0.04), 0 8px 24px -12px rgba(28, 38, 32, 0.12);
}

html, body, [class*="css"] { font-family: 'Inter', sans-serif; }

.stApp { background: var(--bg); color: var(--text); }

#MainMenu, footer { visibility: hidden; }

.block-container {
    padding-top: 1.25rem !important;
    padding-bottom: 7rem !important;
    max-width: 880px !important;
}

/* ── Header ── */
.cr-header { text-align: center; padding: 1.25rem 0 0.25rem; }
.cr-eyebrow {
    font-size: 0.72rem; font-weight: 600; letter-spacing: 3px;
    text-transform: uppercase; color: var(--accent-cool); margin-bottom: 0.5rem;
}
.cr-title {
    font-family: 'Fraunces', serif; font-weight: 600; font-size: 3rem;
    color: var(--text); margin: 0; letter-spacing: -0.5px; line-height: 1.05;
}
.cr-tagline {
    font-size: 0.95rem; color: var(--text-muted); margin: 0.7rem auto 0;
    max-width: 540px; line-height: 1.5;
}
.cr-stripe { height: 6px; border-radius: 6px; background: var(--stripes); margin: 1.4rem auto 0; max-width: 320px; }

/* ── Section labels ── */
.cr-section-label {
    font-size: 0.72rem; font-weight: 700; letter-spacing: 2.5px;
    text-transform: uppercase; color: var(--text-muted);
    margin: 2.2rem 0 0.75rem; display: flex; align-items: center; gap: 0.5rem;
}
.cr-section-label:first-of-type { margin-top: 0.5rem; }

/* ── Did you know card ── */
.cr-fact-card {
    background: var(--surface); border: 1px solid var(--border);
    border-left: 4px solid var(--accent-warm); border-radius: var(--radius-lg);
    padding: 1.15rem 1.4rem; box-shadow: var(--shadow-card);
}
.cr-fact-text { font-family: 'Fraunces', serif; font-size: 1.05rem; line-height: 1.55; color: var(--text); }
.cr-fact-source {
    font-size: 0.74rem; color: var(--text-muted); margin-top: 0.65rem;
    font-family: 'IBM Plex Mono', monospace; letter-spacing: 0.2px;
}

/* ── Buttons (global baseline) ── */
.stButton > button {
    border-radius: var(--radius-md) !important;
    font-family: 'Inter', sans-serif !important;
    font-weight: 500 !important;
    border: 1px solid var(--border) !important;
    color: var(--text) !important;
    transition: all 0.15s ease !important;
}
.stButton > button:hover {
    border-color: var(--accent-cool) !important;
    color: var(--accent-cool) !important;
    background: var(--accent-cool-soft) !important;
}
.stButton > button[kind="primary"] {
    background: var(--accent-cool) !important;
    border-color: var(--accent-cool) !important;
    color: #ffffff !important;
}
.stButton > button[kind="primary"]:hover { background: #124a52 !important; }
.stButton > button:disabled { opacity: 0.4 !important; }

/* ── Suggested-question chips ── */
.st-key-chip_row .stButton > button {
    white-space: normal; text-align: left; height: 100%; width: 100%;
    font-size: 0.82rem; line-height: 1.35; padding: 0.7rem 0.9rem;
    background: var(--surface);
}

/* ── Clear conversation (ghost button) ── */
.st-key-clear_btn .stButton > button {
    border-color: transparent !important; background: transparent !important;
    color: var(--text-muted) !important; font-size: 0.8rem !important;
    padding: 0.3rem 0.7rem !important;
}
.st-key-clear_btn .stButton > button:hover {
    color: var(--error) !important; border-color: var(--error) !important;
    background: rgba(179, 38, 30, 0.06) !important;
}

/* ── New fact (ghost button) ── */
.st-key-fact_btn .stButton > button {
    border-color: transparent !important; background: transparent !important;
    color: var(--accent-warm) !important; font-size: 0.78rem !important;
}
.st-key-fact_btn .stButton > button:hover {
    background: var(--accent-warm-soft) !important; border-color: transparent !important;
    color: var(--accent-warm) !important;
}

/* ── Empty-state hint ── */
.cr-empty-hint {
    text-align: center; color: var(--text-muted); font-size: 0.85rem;
    margin: 1.6rem 0; font-style: italic;
}

/* ── Chat bubbles ── */
[data-testid="stChatMessage"] {
    background: var(--surface); border: 1px solid var(--border);
    border-radius: var(--radius-lg); box-shadow: var(--shadow-card);
    margin-bottom: 0.65rem;
}
[data-testid="stChatInput"] { border-radius: var(--radius-lg) !important; }
[data-testid="stChatInput"] textarea { font-family: 'Inter', sans-serif !important; }

/* ── Sources ── */
.cr-source {
    background: var(--surface-muted); border: 1px solid var(--border);
    border-radius: var(--radius-sm); padding: 0.55rem 0.85rem; margin-bottom: 0.4rem;
}
.cr-source-name { font-size: 0.82rem; font-weight: 600; color: var(--accent-cool); }
.cr-source-meta {
    font-size: 0.72rem; color: var(--text-muted); margin-top: 0.15rem;
    font-family: 'IBM Plex Mono', monospace;
}
.streamlit-expanderHeader { font-size: 0.85rem !important; color: var(--text-muted) !important; }

/* ── Sidebar / Quiz ── */
section[data-testid="stSidebar"] {
    background: var(--surface-muted); border-right: 1px solid var(--border);
}
.cr-quiz-title { font-family: 'Fraunces', serif; font-size: 1.35rem; color: var(--text); margin-bottom: 0.15rem; }
.cr-quiz-sub { font-size: 0.78rem; color: var(--text-muted); margin-bottom: 1.1rem; line-height: 1.4; }
.cr-quiz-question { font-weight: 600; font-size: 0.95rem; margin: 0.7rem 0 0.6rem; color: var(--text); line-height: 1.4; }
.cr-quiz-score-big {
    font-family: 'IBM Plex Mono', monospace; font-size: 2.1rem; font-weight: 500;
    color: var(--accent-cool); text-align: center; margin: 0.5rem 0;
}
.st-key-quiz_panel [data-testid="stProgress"] > div > div { background: var(--stripes) !important; }
.st-key-quiz_panel .stCaption { font-family: 'IBM Plex Mono', monospace !important; }

/* ── Footer ── */
.cr-footer {
    text-align: center; color: var(--text-muted); font-size: 0.75rem;
    padding: 2rem 0 1rem; margin-top: 2.5rem; border-top: 1px solid var(--border); line-height: 1.6;
}

::-webkit-scrollbar { width: 6px; }
::-webkit-scrollbar-track { background: transparent; }
::-webkit-scrollbar-thumb { background: var(--border); border-radius: 4px; }
</style>
""", unsafe_allow_html=True)

# ─── CONTENU : LE SAVIEZ-VOUS ? ──────────────────────────────────────────────
FACTS = [
    {"fact": "Chaque seconde, l'équivalent d'un terrain de football de forêt disparaît sur Terre. Pourtant, un arbre mature peut absorber jusqu'à 22 kg de CO2 par an.", "source": "FAO, 2023"},
    {"fact": "La fonte des glaciers alpins s'accélère : ils ont perdu la moitié de leur volume depuis 1900. À ce rythme, les Alpes pourraient perdre 90 % de leurs glaciers d'ici 2100.", "source": "GIEC AR6, WGII"},
    {"fact": "Les canicules tuent plus de personnes chaque année que tous les autres phénomènes météorologiques extrêmes combinés : ouragans, inondations, sécheresses et tornades réunis.", "source": "Organisation météorologique mondiale, 2021"},
    {"fact": "Un vol Paris–New York aller-retour émet environ 1 tonne de CO2 par passager, soit l'équivalent de ce qu'un arbre absorbe en 50 ans.", "source": "MyClimate Calculator"},
    {"fact": "L'océan a absorbé 90 % de la chaleur supplémentaire générée par les activités humaines depuis 1970. Sans lui, la température terrestre aurait déjà dépassé +3 °C.", "source": "GIEC AR6, WGI"},
    {"fact": "En 2023, la température moyenne mondiale a dépassé +1,5 °C par rapport à l'ère préindustrielle sur une année entière — un seuil que l'on pensait atteindre seulement vers 2040.", "source": "Copernicus Climate Change Service, 2024"},
    {"fact": "Les vagues de chaleur océaniques, autrefois rares, sont devenues 20 fois plus fréquentes depuis 1980. Elles menacent des milliards d'organismes marins et les récifs coralliens.", "source": "Nature Climate Change, 2022"},
    {"fact": "Produire un kilo de bœuf génère environ 60 kg d'équivalent CO2 — vingt fois plus qu'un kilo de légumes. Réduire sa consommation de viande de moitié diviserait par deux son empreinte alimentaire.", "source": "Poore & Nemecek, Science 2018"},
    {"fact": "D'ici 2050, les villes européennes pourraient connaître jusqu'à +8 °C en été à cause de l'effet d'îlot de chaleur urbain : le béton retient la chaleur la nuit.", "source": "Météo-France / CNRS"},
    {"fact": "Le permafrost sibérien renferme environ 1 500 milliards de tonnes de carbone, près du double des émissions mondiales actuelles. Son dégel pourrait libérer du méthane, 80 fois plus puissant que le CO2 sur 20 ans.", "source": "Nature Geoscience, 2023"},
    {"fact": "Les inondations pourraient coûter 48 milliards d'euros par an en Europe d'ici 2050, contre 7,8 milliards aujourd'hui — alors que seuls 35 % des Européens sont assurés contre ce risque.", "source": "Agence européenne pour l'environnement, 2024"},
    {"fact": "Si chaque foyer européen remplaçait ses ampoules par des LED, l'électricité économisée équivaudrait à la production annuelle de dix centrales nucléaires.", "source": "IEA, Global Energy Review"},
    {"fact": "La saison des feux de forêt dure aujourd'hui 75 jours de plus qu'en 1970 en Californie. En Australie, l'été 2019-2020 a brûlé une surface équivalente à la moitié de la France.", "source": "CalFire / CSIRO"},
    {"fact": "D'ici 2050, entre 50 et 200 millions de personnes pourraient être déplacées par les effets du changement climatique.", "source": "Banque mondiale, rapport Groundswell"},
    {"fact": "L'empreinte carbone moyenne d'un Français est d'environ 9 tonnes de CO2 par an. L'Accord de Paris impliquerait de la ramener à 2 tonnes — la moitié provient des transports et de l'alimentation.", "source": "Haut Conseil pour le Climat, 2024"},
]

# ─── CONTENU : QUESTIONS SUGGÉRÉES ───────────────────────────────────────────
SUGGESTED_QUESTIONS = [
    "Pourquoi les canicules deviennent-elles plus fréquentes ?",
    "Quel est le rôle du changement climatique dans les vagues de chaleur en Europe ?",
    "Quels sont les impacts des canicules sur la santé humaine ?",
    "Qu'est-ce que le jet-stream et comment influence-t-il la météo en Europe ?",
    "Quel budget carbone reste-t-il pour limiter le réchauffement à 1,5 °C ?",
]

# ─── CONTENU : QUIZ ───────────────────────────────────────────────────────────
QUIZ_QUESTIONS = [
    {
        "question": "Quelle est la principale cause du réchauffement climatique observé depuis le milieu du 20e siècle ?",
        "options": ["Les variations naturelles du soleil", "Les activités humaines (combustion d'énergies fossiles)", "Les éruptions volcaniques", "Le cycle naturel de la Terre"],
        "correct": 1,
        "explanation": "Le GIEC est catégorique : le réchauffement observé depuis 1950 est dû, sans équivoque, aux activités humaines — en premier lieu à la combustion d'énergies fossiles.",
    },
    {
        "question": "Quel gaz à effet de serre est le principal responsable du réchauffement d'origine humaine ?",
        "options": ["Le méthane (CH4)", "Le dioxyde de carbone (CO2)", "L'ozone (O3)", "La vapeur d'eau"],
        "correct": 1,
        "explanation": "Le CO2, émis surtout par la combustion du charbon, du pétrole et du gaz, est le principal gaz responsable du réchauffement d'origine humaine.",
    },
    {
        "question": "Selon le GIEC (AR6), de combien la température mondiale a-t-elle déjà augmenté par rapport à l'ère préindustrielle ?",
        "options": ["Environ 0,5 °C", "Environ 1,1 °C", "Environ 2,5 °C", "Environ 4 °C"],
        "correct": 1,
        "explanation": "L'AR6 estime le réchauffement à environ 1,1 °C entre la période 2011-2020 et la période 1850-1900.",
    },
    {
        "question": "Quelle part de l'excès de chaleur générée par les activités humaines les océans ont-ils absorbée depuis 1970 ?",
        "options": ["Environ 25 %", "Environ 50 %", "Environ 90 %", "Environ 10 %"],
        "correct": 2,
        "explanation": "Les océans jouent un rôle de régulateur thermique majeur : environ 90 % de la chaleur excédentaire y a été absorbée depuis 1970.",
    },
    {
        "question": "Qu'est-ce que le jet-stream ?",
        "options": ["Un courant marin chaud", "Un puissant courant atmosphérique en haute altitude", "Un type de nuage de canicule", "Une mesure de la pollution de l'air"],
        "correct": 1,
        "explanation": "Le jet-stream est un puissant courant d'air en haute altitude qui guide les systèmes météorologiques. Son affaiblissement est associé à des vagues de chaleur plus persistantes en Europe.",
    },
    {
        "question": "Quel est l'objectif central de l'Accord de Paris (2015) ?",
        "options": ["Limiter le réchauffement à 1,5–2 °C par rapport à l'ère préindustrielle", "Réduire la population mondiale", "Interdire toute énergie fossile d'ici 2030", "Stabiliser le prix du pétrole"],
        "correct": 0,
        "explanation": "L'Accord de Paris vise à contenir la hausse de la température moyenne mondiale nettement en dessous de 2 °C, en poursuivant les efforts pour la limiter à 1,5 °C.",
    },
    {
        "question": "Que signifie le sigle GIEC ?",
        "options": ["Groupe international des écologistes certifiés", "Groupe d'experts intergouvernemental sur l'évolution du climat", "Gestion internationale des émissions de carbone", "Groupe indépendant d'étude du climat"],
        "correct": 1,
        "explanation": "Le GIEC (IPCC en anglais) est l'organisme scientifique des Nations Unies chargé d'évaluer les connaissances sur le changement climatique.",
    },
    {
        "question": "Quelle part de l'empreinte carbone d'un Français vient en moyenne des transports et de l'alimentation ?",
        "options": ["Environ 10 %", "Environ 25 %", "Environ 50 %", "Environ 90 %"],
        "correct": 2,
        "explanation": "Selon le Haut Conseil pour le Climat, transports et alimentation représentent à eux deux environ la moitié de l'empreinte carbone moyenne d'un Français (~9 tonnes de CO2/an).",
    },
]

# ─── SESSION STATE ────────────────────────────────────────────────────────────
_defaults = {
    "messages": [],
    "rag": None,
    "did_you_know_idx": random.randint(0, len(FACTS) - 1),
    "quiz_index": 0,
    "quiz_score": 0,
    "quiz_answered": False,
    "quiz_selected": None,
    "quiz_finished": False,
}
for _k, _v in _defaults.items():
    if _k not in st.session_state:
        st.session_state[_k] = _v


def reset_quiz():
    st.session_state.quiz_index = 0
    st.session_state.quiz_score = 0
    st.session_state.quiz_answered = False
    st.session_state.quiz_selected = None
    st.session_state.quiz_finished = False


# ─── INIT RAG (mis en cache) ──────────────────────────────────────────────────
@st.cache_resource(show_spinner=False)
def load_rag():
    api_key = os.getenv("GROQ_API_KEY")
    if not api_key:
        return None, (
            f"Clé API Groq manquante. Vérifiez qu'un fichier nommé exactement .env "
            f"(pas .env.txt) existe à cet emplacement : {ENV_PATH} et qu'il contient "
            f"la ligne GROQ_API_KEY=votre_clé (sans guillemets, sans espace)."
        )

    index_path = Path(FAISS_INDEX_PATH)
    if not (index_path.with_suffix(".faiss").exists() and index_path.with_suffix(".pkl").exists()):
        return None, "Index FAISS non trouvé. Lancez d'abord : python scripts/ingest.py"

    try:
        embedder = Embedder()
        vector_store = VectorStore(dimension=embedder.dimension)
        vector_store.load(FAISS_INDEX_PATH)
        reranker = Reranker()
        llm = GroqLLM(api_key=api_key)

        rag = ClimateRAG(embedder=embedder, vector_store=vector_store, reranker=reranker, llm=llm)
        return rag, None
    except Exception as e:
        return None, str(e)


# ─── EN-TÊTE ──────────────────────────────────────────────────────────────────
st.markdown("""
<div class="cr-header">
    <div class="cr-eyebrow">Assistant scientifique</div>
    <h1 class="cr-title">ClimateRAG</h1>
    <p class="cr-tagline">Posez vos questions sur le réchauffement climatique, les canicules et les
    événements extrêmes — réponses fondées sur les rapports du GIEC, Copernicus et Météo-France.</p>
    <div class="cr-stripe"></div>
</div>
""", unsafe_allow_html=True)

# ─── SIDEBAR : QUIZ (accessible sur le côté) ─────────────────────────────────
with st.sidebar:
    with st.container(key="quiz_panel"):
        st.markdown('<div class="cr-quiz-title">🧠 Quiz Climat</div>', unsafe_allow_html=True)
        st.markdown('<div class="cr-quiz-sub">Testez vos connaissances sur le changement climatique en 8 questions.</div>', unsafe_allow_html=True)

        total = len(QUIZ_QUESTIONS)

        if not st.session_state.quiz_finished:
            idx = st.session_state.quiz_index
            q = QUIZ_QUESTIONS[idx]

            st.progress(idx / total)
            st.caption(f"Question {idx + 1} / {total}  ·  Score {st.session_state.quiz_score}")
            st.markdown(f'<div class="cr-quiz-question">{q["question"]}</div>', unsafe_allow_html=True)

            if not st.session_state.quiz_answered:
                placeholder = "— Choisissez une réponse —"
                choice = st.radio(
                    "Réponse",
                    [placeholder] + q["options"],
                    key=f"quiz_radio_{idx}",
                    label_visibility="collapsed",
                )
                ready = choice != placeholder
                if st.button("Valider", key="quiz_validate", type="primary", use_container_width=True, disabled=not ready):
                    st.session_state.quiz_selected = choice
                    st.session_state.quiz_answered = True
                    if choice == q["options"][q["correct"]]:
                        st.session_state.quiz_score += 1
                    st.rerun()
            else:
                selected = st.session_state.quiz_selected
                correct_answer = q["options"][q["correct"]]
                if selected == correct_answer:
                    st.success(f"Bonne réponse — {selected}")
                else:
                    st.error(f"Pas tout à fait. Réponse : {correct_answer}")
                st.caption(q["explanation"])

                is_last = idx + 1 >= total
                label = "Voir mon score →" if is_last else "Question suivante →"
                if st.button(label, key="quiz_next", type="primary", use_container_width=True):
                    if is_last:
                        st.session_state.quiz_finished = True
                    else:
                        st.session_state.quiz_index += 1
                        st.session_state.quiz_answered = False
                        st.session_state.quiz_selected = None
                    st.rerun()
        else:
            score = st.session_state.quiz_score
            pct = round(100 * score / total)
            if pct >= 80:
                msg = "🌟 Expert du climat !"
            elif pct >= 50:
                msg = "🌱 Bien joué, continuez comme ça !"
            else:
                msg = "📚 Discutez avec l'assistant pour en apprendre plus."
            st.markdown(f'<div class="cr-quiz-score-big">{score}/{total}</div>', unsafe_allow_html=True)
            st.markdown(f'<p style="text-align:center;color:var(--text-muted);font-size:0.85rem;">{msg}</p>', unsafe_allow_html=True)
            if st.button("Recommencer le quiz", key="quiz_restart", use_container_width=True):
                reset_quiz()
                st.rerun()

# ─── SECTION 1 : LE SAVIEZ-VOUS ? ────────────────────────────────────────────
st.markdown('<div class="cr-section-label">🌡️ Le saviez-vous ?</div>', unsafe_allow_html=True)

fact = FACTS[st.session_state.did_you_know_idx]
st.markdown(f"""
<div class="cr-fact-card">
    <div class="cr-fact-text">{fact['fact']}</div>
    <div class="cr-fact-source">Source · {fact['source']}</div>
</div>
""", unsafe_allow_html=True)

_, fact_btn_col = st.columns([6, 1.4])
with fact_btn_col:
    with st.container(key="fact_btn"):
        if st.button("↻ Autre fait", key="new_fact", use_container_width=True):
            st.session_state.did_you_know_idx = random.randint(0, len(FACTS) - 1)
            st.rerun()

# ─── SECTION 2 : DISCUTER AVEC L'ASSISTANT ───────────────────────────────────
label_col, clear_col = st.columns([5, 1.4])
with label_col:
    st.markdown('<div class="cr-section-label">💬 Discuter avec l\'assistant</div>', unsafe_allow_html=True)
with clear_col:
    with st.container(key="clear_btn"):
        if st.button("Effacer", key="clear_history", use_container_width=True):
            st.session_state.messages = []
            st.rerun()

st.caption("Questions fréquentes")
new_prompt = None
with st.container(key="chip_row"):
    chip_cols = st.columns(len(SUGGESTED_QUESTIONS))
    for i, q_text in enumerate(SUGGESTED_QUESTIONS):
        with chip_cols[i]:
            if st.button(q_text, key=f"chip_{i}", use_container_width=True):
                new_prompt = q_text

if not st.session_state.messages:
    st.markdown(
        '<p class="cr-empty-hint">👋 Choisissez une question ci-dessus ou écrivez la vôtre pour commencer.</p>',
        unsafe_allow_html=True,
    )

# ─── HISTORIQUE DE CONVERSATION ───────────────────────────────────────────────
for msg in st.session_state.messages:
    avatar = "🌍" if msg["role"] == "assistant" else None
    with st.chat_message(msg["role"], avatar=avatar):
        st.markdown(msg["content"])
        if msg.get("sources"):
            with st.expander(f"📚 Sources utilisées ({len(msg['sources'])})"):
                for src in msg["sources"]:
                    st.markdown(f"""
<div class="cr-source">
    <div class="cr-source-name">{src['source']} — page {src['page']}</div>
    <div class="cr-source-meta">FAISS {src['faiss_score']:.3f} · Rerank {src['rerank_score']:.3f}</div>
</div>
""", unsafe_allow_html=True)

# ─── ZONE DE SAISIE (toujours ancrée en bas) ─────────────────────────────────
typed = st.chat_input("Posez votre question sur le climat…")
if typed:
    new_prompt = typed

# ─── TRAITEMENT DE LA QUESTION ────────────────────────────────────────────────
if new_prompt:
    st.session_state.messages.append({"role": "user", "content": new_prompt})
    with st.chat_message("user"):
        st.markdown(new_prompt)

    answer, sources = None, []
    with st.chat_message("assistant", avatar="🌍"):
        if st.session_state.rag is None:
            with st.spinner("Initialisation du pipeline…"):
                rag, error = load_rag()
            if error:
                st.error(f"Erreur : {error}")
                st.stop()
            st.session_state.rag = rag

        with st.spinner("Recherche dans les rapports scientifiques…"):
            try:
                result = st.session_state.rag.ask(new_prompt)
                answer = result["answer"]
                sources = result.get("sources", [])
            except Exception as e:
                st.error(f"Erreur : {e}")

        if answer:
            st.markdown(answer)
            if sources:
                with st.expander(f"📚 Sources utilisées ({len(sources)})"):
                    for src in sources:
                        st.markdown(f"""
<div class="cr-source">
    <div class="cr-source-name">{src['source']} — page {src['page']}</div>
    <div class="cr-source-meta">FAISS {src['faiss_score']:.3f} · Rerank {src['rerank_score']:.3f}</div>
</div>
""", unsafe_allow_html=True)

    if answer:
        st.session_state.messages.append({"role": "assistant", "content": answer, "sources": sources})
        st.session_state.did_you_know_idx = random.randint(0, len(FACTS) - 1)

# ─── FOOTER ───────────────────────────────────────────────────────────────────
st.markdown("""
<div class="cr-footer">
    ClimateRAG · Réponses fondées sur les rapports du GIEC (AR6), Copernicus et Météo-France<br>
    Les réponses sont générées automatiquement — vérifiez toujours les sources originales.
</div>
""", unsafe_allow_html=True)