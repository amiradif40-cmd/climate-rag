import streamlit as st
import sys
from pathlib import Path
import os
import random

# --- PATH SETUP ---
PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT / "src"))

# --- SECRETS ---
from dotenv import load_dotenv
ENV_PATH = PROJECT_ROOT / ".env"
if ENV_PATH.exists():
    load_dotenv(dotenv_path=ENV_PATH)

try:
    GROQ_API_KEY = st.secrets.get("GROQ_API_KEY") or os.getenv("GROQ_API_KEY")
except Exception:
    GROQ_API_KEY = os.getenv("GROQ_API_KEY")

FAISS_INDEX_PATH = str(PROJECT_ROOT / "data" / "index" / "climate_index")

# --- IMPORTS RAG ---
try:
    from embeddings.embedder import Embedder
    from retrieval.vector_store import VectorStore
    from retrieval.reranker import Reranker
    from generation.llm import GroqLLM
    from pipeline.rag_pipeline import ClimateRAG
    RAG_AVAILABLE = True
except ImportError as e:
    RAG_AVAILABLE = False
    st.error(f"Erreur d'import RAG : {e}")

# --- PAGE CONFIG ---
st.set_page_config(
    page_title="ClimateRAG",
    layout="wide",
    initial_sidebar_state="collapsed",
)

# --- DESIGN SYSTEM ---
st.markdown("""
<style>
@import url('https://fonts.googleapis.com/css2?family=Fraunces:opsz,wght@9..144,400;9..144,500;9..144,600;9..144,700&family=Inter:wght@400;500;600;700&family=IBM+Plex+Mono:wght@400;500&display=swap');

:root {
    --bg: #F6F7F4; --surface: #FFFFFF; --surface-muted: #EDF1EA;
    --border: #E1E6DD; --text: #1C2620; --text-muted: #64705F;
    --accent-cool: #175E68; --accent-cool-soft: rgba(23, 94, 104, 0.08);
    --accent-warm: #C2410C; --accent-warm-soft: rgba(194, 65, 12, 0.08);
    --success: #2F9E5C; --error: #B3261E;
    --stripes: linear-gradient(90deg, #08306B 0%, #2166AC 16%, #92C5DE 32%, #F2F1EA 50%, #F4A582 68%, #B2182B 84%, #67001F 100%);
    --radius-lg: 18px; --radius-md: 12px; --radius-sm: 8px;
    --shadow-card: 0 1px 2px rgba(28, 38, 32, 0.04), 0 8px 24px -12px rgba(28, 38, 32, 0.12);
}

/* === FIX LISIBILITE GLOBAL - MODE SOMBRE TELEPHONE === */
html { color-scheme: light !important; }
.stApp, .stApp * {
    color: #1C2620 !important;
    -webkit-text-fill-color: #1C2620 !important;
}
.stButton > button[kind="primary"],
.stButton > button[kind="primary"] * {
    color: #ffffff !important;
    -webkit-text-fill-color: #ffffff !important;
}
a, a:visited, a:hover {
    color: var(--accent-cool) !important;
    -webkit-text-fill-color: var(--accent-cool) !important;
}
[data-testid="stChatInput"] {
    background: var(--surface) !important;
}
[data-testid="stChatInput"] textarea {
    background: #FFFFFF !important;
    color: #1C2620 !important;
    -webkit-text-fill-color: #1C2620 !important;
    caret-color: #1C2620 !important;
}
[data-testid="stChatInput"] textarea::placeholder {
    color: #64705F !important;
    -webkit-text-fill-color: #64705F !important;
    opacity: 1 !important;
}
[data-testid="stChatMessage"] {
    background: var(--surface) !important;
}
[data-testid="stAlert"] * {
    color: #1C2620 !important;
    -webkit-text-fill-color: #1C2620 !important;
    opacity: 1 !important;
}
div[role="radiogroup"] label,
div[role="radiogroup"] label * {
    color: #1C2620 !important;
    -webkit-text-fill-color: #1C2620 !important;
    opacity: 1 !important;
}

html, body, [class*="css"] { font-family: 'Inter', sans-serif; }
.stApp { background: var(--bg); color: var(--text); }
#MainMenu, footer { visibility: hidden; }
.block-container { padding-top: 1rem !important; padding-bottom: 5rem !important; max-width: 880px !important; }

/* --- MOBILE FIXES --- */
@media (max-width: 768px) {
    .block-container { padding: 0.5rem 1rem !important; max-width: 100% !important; }
    .cr-title { font-size: 2rem !important; }
    .cr-tagline { font-size: 0.85rem !important; max-width: 100% !important; padding: 0 0.5rem; }
    .cr-fact-card { padding: 0.8rem 1rem !important; }
    .cr-fact-text { font-size: 0.95rem !important; }
    .cr-section-label { margin: 1.5rem 0 0.5rem !important; }
    [data-testid="stChatMessage"] { padding: 0.5rem !important; margin-bottom: 0.4rem !important; }
    [data-testid="stChatMessage"] > div:first-child { display: none !important; }
    [data-testid="stChatInput"] { padding: 0.5rem !important; }
    section[data-testid="stSidebar"] { width: 100% !important; }
    [data-testid="stChatInput"] textarea {
        background: #FFFFFF !important;
        color: #1C2620 !important;
        -webkit-text-fill-color: #1C2620 !important;
    }
}

.cr-header { text-align: center; padding: 1rem 0 0.25rem; }
.cr-eyebrow { font-size: 0.72rem; font-weight: 600; letter-spacing: 3px; text-transform: uppercase; color: var(--accent-cool); margin-bottom: 0.5rem; }
.cr-title { font-family: 'Fraunces', serif; font-weight: 600; font-size: 3rem; color: var(--text); margin: 0; letter-spacing: -0.5px; line-height: 1.05; }
.cr-tagline { font-size: 0.95rem; color: var(--text-muted); margin: 0.7rem auto 0; max-width: 540px; line-height: 1.5; padding: 0 1rem; }
.cr-stripe { height: 6px; border-radius: 6px; background: var(--stripes); margin: 1.4rem auto 0; max-width: 320px; }

.cr-section-label { font-size: 0.72rem; font-weight: 700; letter-spacing: 2.5px; text-transform: uppercase; color: var(--text-muted); margin: 2.2rem 0 0.75rem; display: flex; align-items: center; gap: 0.5rem; }
.cr-section-label:first-of-type { margin-top: 0.5rem; }

.cr-fact-card { background: var(--surface); border: 1px solid var(--border); border-left: 4px solid var(--accent-warm); border-radius: var(--radius-lg); padding: 1.15rem 1.4rem; box-shadow: var(--shadow-card); }
.cr-fact-text { font-family: 'Fraunces', serif; font-size: 1.05rem; line-height: 1.55; color: var(--text); }
.cr-fact-source { font-size: 0.74rem; color: var(--text-muted); margin-top: 0.65rem; font-family: 'IBM Plex Mono', monospace; letter-spacing: 0.2px; }

.stButton > button { border-radius: var(--radius-md) !important; font-family: 'Inter', sans-serif !important; font-weight: 500 !important; border: 1px solid var(--border) !important; color: var(--text) !important; transition: all 0.15s ease !important; }
.stButton > button:hover { border-color: var(--accent-cool) !important; color: var(--accent-cool) !important; background: var(--accent-cool-soft) !important; }
.stButton > button[kind="primary"] { background: var(--accent-cool) !important; border-color: var(--accent-cool) !important; color: #ffffff !important; }
.stButton > button[kind="primary"]:hover { background: #124a52 !important; }
.stButton > button:disabled { opacity: 0.4 !important; }

.st-key-chip_row .stButton > button { white-space: normal; text-align: left; height: 100%; width: 100%; font-size: 0.82rem; line-height: 1.35; padding: 0.7rem 0.9rem; background: var(--surface); }
.st-key-clear_btn .stButton > button { border-color: transparent !important; background: transparent !important; color: var(--text-muted) !important; font-size: 0.8rem !important; padding: 0.3rem 0.7rem !important; }
.st-key-clear_btn .stButton > button:hover { color: var(--error) !important; border-color: var(--error) !important; background: rgba(179, 38, 30, 0.06) !important; }
.st-key-fact_btn .stButton > button { border-color: transparent !important; background: transparent !important; color: var(--accent-warm) !important; font-size: 0.78rem !important; }
.st-key-fact_btn .stButton > button:hover { background: var(--accent-warm-soft) !important; border-color: transparent !important; color: var(--accent-warm) !important; }

.cr-empty-hint { text-align: center; color: var(--text-muted); font-size: 0.85rem; margin: 1.6rem 0; font-style: italic; }

[data-testid="stChatMessage"] { background: var(--surface); border: 1px solid var(--border); border-radius: var(--radius-lg); box-shadow: var(--shadow-card); margin-bottom: 0.65rem; padding: 0.75rem 1rem; }
[data-testid="stChatMessage"] > div:first-child { display: none !important; }
[data-testid="stChatInput"] { border-radius: var(--radius-lg) !important; }
[data-testid="stChatInput"] textarea { font-family: 'Inter', sans-serif !important; }

.cr-source { background: var(--surface-muted); border: 1px solid var(--border); border-radius: var(--radius-sm); padding: 0.55rem 0.85rem; margin-bottom: 0.4rem; }
.cr-source-name { font-size: 0.82rem; font-weight: 600; color: var(--accent-cool); }
.cr-source-meta { font-size: 0.72rem; color: var(--text-muted); margin-top: 0.15rem; font-family: 'IBM Plex Mono', monospace; }
.streamlit-expanderHeader { font-size: 0.85rem !important; color: var(--text-muted) !important; }

section[data-testid="stSidebar"] { background: var(--surface-muted); border-right: 1px solid var(--border); }
.cr-quiz-title { font-family: 'Fraunces', serif; font-size: 1.35rem; color: var(--text); margin-bottom: 0.15rem; }
.cr-quiz-sub { font-size: 0.78rem; color: var(--text-muted); margin-bottom: 1.1rem; line-height: 1.4; }
.cr-quiz-question { font-weight: 600; font-size: 0.95rem; margin: 0.7rem 0 0.6rem; color: var(--text); line-height: 1.4; }
.cr-quiz-score-big { font-family: 'IBM Plex Mono', monospace; font-size: 2.1rem; font-weight: 500; color: var(--accent-cool); text-align: center; margin: 0.5rem 0; }
.st-key-quiz_panel [data-testid="stProgress"] > div > div { background: var(--stripes) !important; }
.st-key-quiz_panel .stCaption { font-family: 'IBM Plex Mono', monospace !important; }

.cr-footer { text-align: center; color: var(--text-muted); font-size: 0.75rem; padding: 2rem 0 1rem; margin-top: 2.5rem; border-top: 1px solid var(--border); line-height: 1.6; }

::-webkit-scrollbar { width: 6px; }
::-webkit-scrollbar-track { background: transparent; }
::-webkit-scrollbar-thumb { background: var(--border); border-radius: 4px; }

</style>
""", unsafe_allow_html=True)

# --- CONTENU ---
FACTS = [
    {"fact": "Le climat, c'est la meteo sur le long terme. Une vague de froid en hiver ne contredit pas le rechauffement climatique global.", "source": "GIEC AR6, WGI"},
    {"fact": "La chaleur extreme est l'un des phenomenes meteorologiques les plus meurtriers. Les canicules peuvent causer de nombreuses victimes, souvent de maniere indirecte (aggravation de maladies cardiaques, respiratoires, renales).", "source": "OMM / Sante publique France"},
    {"fact": "L'ocean a absorbe environ 90 % de la chaleur supplementaire generee par les activites humaines depuis les annees 1970.", "source": "GIEC AR6, WGI"},
    {"fact": "En 2023, la temperature mondiale annuelle a atteint environ +1,48 C par rapport a l'ere preindustrielle, et 2024 a depasse +1,5 C. Ces depassements temporaires ne signifient pas que l'objectif de long terme de l'Accord de Paris est deja perdu, mais montrent l'acceleration du rechauffement.", "source": "Copernicus C3S / OMM, 2024-2025"},
    {"fact": "Les vagues de chaleur oceaniques sont devenues nettement plus frequentes et plus intenses depuis les annees 1980. Elles menacent les recifs coralliens et de nombreux ecosystemes marins.", "source": "GIEC AR6, WGI"},
    {"fact": "Une journee de canicule en ville peut etre 2 a 8 C plus chaude qu'a la campagne a cause de l'ilot de chaleur urbain. Le beton et l'asphalte emmagasinent la chaleur la nuit.", "source": "GIEC AR6, WGII"},
    {"fact": "Le jet-stream est un puissant courant d'air en haute altitude qui guide les depressions et les anticyclones. Son ralentissement peut bloquer les vagues de chaleur sur une meme region.", "source": "Meteo-France / CNRS"},
    {"fact": "Les glaciers alpins ont perdu une grande partie de leur masse depuis 1900. Leur fonte s'accelere avec le rechauffement.", "source": "GIEC AR6, WGI"},
    {"fact": "Le permafrost siberien et arctique renferme d'importantes reserves de carbone. Son rechauffement pourrait liberer du CO2 et du methane, amplifiant le rechauffement.", "source": "GIEC AR6, WGI"},
    {"fact": "L'humidite joue un role cle dans les canicules : a temperature identique, un air sec est moins dangereux qu'un air humide, car la sueur s'evapore mal quand l'air est deja sature.", "source": "Meteo-France / OMM"},
    {"fact": "La glace de mer arctique fond en ete, mais cela ne fait pas monter le niveau des oceans (comme la glace dans un verre). En revanche, la fonte des glaciers terrestres et des calottes, si.", "source": "GIEC AR6, WGI"},
    {"fact": "Le rechauffement climatique ne se traduit pas seulement par des temperatures plus chaudes : il deplace aussi la distribution des temperatures, rendant les extremes plus frequents.", "source": "GIEC AR6, WGI"},
    {"fact": "Les feux de foret de grande ampleur, comme en Australie en 2019-2020 ou au Canada en 2023, sont intensifies par les conditions chaudes et seches liees au changement climatique.", "source": "GIEC AR6, WGII"},
    {"fact": "L'Accord de Paris (2015) vise a contenir le rechauffement 'nettement en dessous de 2 C' et a poursuivre les efforts pour le limiter a 1,5 C. Ce sont des objectifs politiques, pas des quotas individuels.", "source": "Accord de Paris / GIEC"},
    {"fact": "Le GIEC estime que le rechauffement observe entre 2011-2020 et 1850-1900 est d'environ 1,1 C. Il est 'sans equivoque' que ce rechauffement est du aux activites humaines.", "source": "GIEC AR6, WGI"},
    {"fact": "La Terre n'est pas ronde : elle est legerement aplatie aux poles et renflee a l'equateur, comme une orange pressee. Cette forme influence la distribution de la chaleur.", "source": "NASA / geodesie"},
    {"fact": "Si vous etes ne avant 1988, vous n'avez jamais vecu une annee plus froide que la moyenne du 20e siecle. Chaque annee depuis est au-dessus.", "source": "GIEC AR6, WGI / NASA GISS"},
    {"fact": "Les pissenlits poussent mieux dans les villes chaudes qu'a la campagne. L'ilot de chaleur urbain les fait fleurir plus tot et plus gros.", "source": "Etudes d'urbanisation vegetale"},
    {"fact": "Le CO2 que vous expirez en une journee (environ 1 kg) ne contribue pas au rechauffement : il fait partie du cycle naturel du carbone. Le probleme, c'est le carbone fossile sorti du sous-sol apres des millions d'annees.", "source": "GIEC AR6, WGI"},
    {"fact": "Les moustiques aiment le changement climatique. Plus il fait chaud et humide, plus ils se reproduisent vite. Certaines maladies comme le dengue gagnent du terrain vers le nord de l'Europe.", "source": "GIEC AR6, WGII / OMS"},
]

SUGGESTED_QUESTIONS = [
    "Pourquoi les canicules deviennent-elles plus frequentes ?",
    "Quel est le role du changement climatique dans les vagues de chaleur en Europe ?",
    "Quels sont les impacts des canicules sur la sante humaine ?",
    "Qu'est-ce que le jet-stream et comment influence-t-il la meteo en Europe ?",
    "Quel budget carbone reste-t-il pour limiter le rechauffement a 1,5 C ?",
]

QUIZ_QUESTIONS = [
    {
        "question": "Quelle est la principale cause du rechauffement climatique observe depuis le milieu du 20e siecle ?",
        "options": ["Les variations naturelles du soleil", "Les activites humaines (emissions de gaz a effet de serre)", "Les eruptions volcaniques", "Le cycle naturel de la Terre"],
        "correct": 1,
        "explanation": "Le GIEC AR6 est categorique : le rechauffement observe depuis 1950 est 'sans equivoque' du aux activites humaines, principalement la combustion d'energies fossiles."
    },
    {
        "question": "Quel gaz a effet de serre est le principal responsable du rechauffement actuel ?",
        "options": ["Le methane (CH4)", "Le dioxyde de carbone (CO2)", "L'ozone (O3)", "La vapeur d'eau"],
        "correct": 1,
        "explanation": "Le CO2, emis surtout par la combustion du charbon, du petrole et du gaz, est le principal contributeur au rechauffement d'origine humaine."
    },
    {
        "question": "Selon le GIEC (AR6), de combien la temperature mondiale a-t-elle deja augmente par rapport a l'ere preindustrielle ?",
        "options": ["Environ 0,5 C", "Environ 1,1 C", "Environ 2,5 C", "Environ 4 C"],
        "correct": 1,
        "explanation": "L'AR6 estime le rechauffement a environ 1,1 C entre la periode 2011-2020 et la periode de reference 1850-1900."
    },
    {
        "question": "Quelle part de la chaleur supplementaire generee par les activites humaines les oceans ont-ils absorbee depuis les annees 1970 ?",
        "options": ["Environ 25 %", "Environ 50 %", "Environ 90 %", "Environ 10 %"],
        "correct": 2,
        "explanation": "Les oceans jouent un role de regulateur thermique majeur : environ 90 % de la chaleur excedentaire y a ete absorbee."
    },
    {
        "question": "Qu'est-ce que le jet-stream ?",
        "options": ["Un courant marin chaud", "Un puissant courant atmospherique en haute altitude", "Un type de nuage de canicule", "Une mesure de la pollution de l'air"],
        "correct": 1,
        "explanation": "Le jet-stream est un puissant courant d'air en haute altitude qui guide les systemes meteorologiques. Son affaiblissement peut bloquer les vagues de chaleur."
    },
    {
        "question": "Quel est l'objectif central de l'Accord de Paris (2015) ?",
        "options": ["Limiter le rechauffement nettement en dessous de 2 C et poursuivre vers 1,5 C", "Reduire la population mondiale", "Interdire toute energie fossile d'ici 2030", "Stabiliser le prix du petrole"],
        "correct": 0,
        "explanation": "L'Accord de Paris vise a contenir la hausse de la temperature mondiale nettement en dessous de 2 C, en poursuivant les efforts pour la limiter a 1,5 C."
    },
    {
        "question": "Que signifie le sigle GIEC ?",
        "options": ["Groupe international des ecologistes certifies", "Groupe d'experts intergouvernemental sur l'evolution du climat", "Gestion internationale des emissions de carbone", "Groupe independant d'etude du climat"],
        "correct": 1,
        "explanation": "Le GIEC (IPCC en anglais) est l'organisme scientifique des Nations Unies charge d'evaluer les connaissances sur le changement climatique."
    },
    {
        "question": "Pourquoi une canicule en ville peut-elle etre plus dangereuse qu'a la campagne ?",
        "options": ["Parce qu'il y a plus de monde", "A cause de l'ilot de chaleur urbain (beton et asphalte emmagasinent la chaleur)", "Parce que les villes sont plus proches du soleil", "A cause de la pollution uniquement"],
        "correct": 1,
        "explanation": "L'ilot de chaleur urbain peut rendre une journee de canicule 2 a 8 C plus chaude en ville qu'a la campagne. Le beton et l'asphalte emmagasinent la chaleur et la restituent la nuit."
    },
    {
        "question": "Le CO2 que vous expirez en respirant contribue-t-il au rechauffement climatique ?",
        "options": ["Oui, il faut donc moins respirer", "Non, il fait partie du cycle naturel du carbone", "Oui, mais seulement si on mange de la viande", "Non, car le CO2 humain est plus leger que celui des usines"],
        "correct": 1,
        "explanation": "Le CO2 expire par les humains et les animaux fait partie du cycle naturel du carbone (absorbe par les plantes, reemis par la respiration). Le probleme vient du carbone fossile (charbon, petrole, gaz) sorti du sous-sol apres des millions d'annees."
    },
]

# --- SESSION STATE ---
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
    for k, v in _defaults.items():
        if k.startswith("quiz_"):
            st.session_state[k] = v
    st.session_state.quiz_index = 0
    st.session_state.quiz_score = 0
    st.session_state.quiz_answered = False
    st.session_state.quiz_selected = None
    st.session_state.quiz_finished = False


# --- INIT RAG ---
@st.cache_resource(show_spinner=False)
def load_rag():
    if not GROQ_API_KEY:
        return None, "Cle API Groq manquante. Configurez-la dans les secrets Streamlit Cloud ou dans un fichier .env local."
    if not RAG_AVAILABLE:
        return None, "Modules RAG non disponibles. Verifiez que src/ est bien dans le PYTHONPATH."
    
    index_path = Path(FAISS_INDEX_PATH)
    faiss_file = index_path.with_suffix(".faiss")
    pkl_file = index_path.with_suffix(".pkl")
    
    if not (faiss_file.exists() and pkl_file.exists()):
        return None, f"Index FAISS non trouve ({faiss_file.name} / {pkl_file.name}). Lancez d'abord : python scripts/ingest.py"
    
    try:
        embedder = Embedder()
        vector_store = VectorStore(dimension=embedder.dimension)
        vector_store.load(FAISS_INDEX_PATH)
        reranker = Reranker()
        llm = GroqLLM(api_key=GROQ_API_KEY)
        rag = ClimateRAG(embedder=embedder, vector_store=vector_store, reranker=reranker, llm=llm)
        return rag, None
    except Exception as e:
        return None, str(e)


# --- EN-TETE ---
st.markdown("""
<div class="cr-header">
    <div class="cr-eyebrow">Assistant scientifique</div>
    <h1 class="cr-title">ClimateRAG</h1>
    <p class="cr-tagline">Posez vos questions sur le rechauffement climatique, les canicules et les evenements extremes - reponses fondees sur les rapports du GIEC, Copernicus et Meteo-France.</p>
    <div class="cr-stripe"></div>
</div>
""", unsafe_allow_html=True)

# --- SIDEBAR : QUIZ ---
with st.sidebar:
    with st.container(key="quiz_panel"):
        st.markdown('<div class="cr-quiz-title">Quiz Climat</div>', unsafe_allow_html=True)
        st.markdown('<div class="cr-quiz-sub">Testez vos connaissances sur le changement climatique en 9 questions.</div>', unsafe_allow_html=True)
        total = len(QUIZ_QUESTIONS)
        
        if not st.session_state.quiz_finished:
            idx = st.session_state.quiz_index
            q = QUIZ_QUESTIONS[idx]
            st.progress(idx / total)
            st.caption(f"Question {idx + 1} / {total}  -  Score {st.session_state.quiz_score}")
            st.markdown(f'<div class="cr-quiz-question">{q["question"]}</div>', unsafe_allow_html=True)
            
            if not st.session_state.quiz_answered:
                placeholder = "- Choisissez une reponse -"
                choice = st.radio("Reponse", [placeholder] + q["options"], key=f"quiz_radio_{idx}", label_visibility="collapsed")
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
                    st.success(f"Bonne reponse - {selected}")
                else:
                    st.error(f"Pas tout a fait. Reponse : {correct_answer}")
                st.caption(q["explanation"])
                
                is_last = idx + 1 >= total
                label = "Voir mon score ->" if is_last else "Question suivante ->"
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
                msg = "Expert du climat !"
            elif pct >= 50:
                msg = "Bien joue, continuez comme ca !"
            else:
                msg = "Discutez avec l'assistant pour en apprendre plus."
            
            st.markdown(f'<div class="cr-quiz-score-big">{score}/{total}</div>', unsafe_allow_html=True)
            st.markdown(f'<p style="text-align:center;color:var(--text-muted);font-size:0.85rem;">{msg}</p>', unsafe_allow_html=True)
            if st.button("Recommencer le quiz", key="quiz_restart", use_container_width=True):
                reset_quiz()
                st.rerun()

# --- SECTION 1 : LE SAVIEZ-VOUS ? ---
st.markdown('<div class="cr-section-label">Le saviez-vous ?</div>', unsafe_allow_html=True)
fact = FACTS[st.session_state.did_you_know_idx]
st.markdown(f"""
<div class="cr-fact-card">
    <div class="cr-fact-text">{fact['fact']}</div>
    <div class="cr-fact-source">Source - {fact['source']}</div>
</div>
""", unsafe_allow_html=True)
_, fact_btn_col = st.columns([6, 1.4])
with fact_btn_col:
    with st.container(key="fact_btn"):
        if st.button("Autre fait", key="new_fact", use_container_width=True):
            st.session_state.did_you_know_idx = random.randint(0, len(FACTS) - 1)
            st.rerun()

# --- SECTION 2 : DISCUTER ---
label_col, clear_col = st.columns([5, 1.4])
with label_col:
    st.markdown('<div class="cr-section-label">Discuter avec l\'assistant</div>', unsafe_allow_html=True)
with clear_col:
    with st.container(key="clear_btn"):
        if st.button("Effacer", key="clear_history", use_container_width=True):
            st.session_state.messages = []
            st.rerun()

st.caption("Questions frequentes")
new_prompt = None
with st.container(key="chip_row"):
    chip_cols = st.columns(len(SUGGESTED_QUESTIONS))
    for i, q_text in enumerate(SUGGESTED_QUESTIONS):
        with chip_cols[i]:
            if st.button(q_text, key=f"chip_{i}", use_container_width=True):
                new_prompt = q_text

if not st.session_state.messages:
    st.markdown('<p class="cr-empty-hint">Choisissez une question ci-dessus ou ecrivez la votre pour commencer.</p>', unsafe_allow_html=True)

# --- HISTORIQUE ---
for msg in st.session_state.messages:
    with st.chat_message(msg["role"]):
        st.markdown(msg["content"])
        if msg.get("sources"):
            with st.expander(f"Sources utilisees ({len(msg['sources'])})"):
                for src in msg["sources"]:
                    st.markdown(f"""
<div class="cr-source">
    <div class="cr-source-name">{src['source']} - page {src['page']}</div>
    <div class="cr-source-meta">FAISS {src['faiss_score']:.3f} - Rerank {src['rerank_score']:.3f}</div>
</div>
""", unsafe_allow_html=True)

# --- SAISIE ---
typed = st.chat_input("Posez votre question sur le climat...")
if typed:
    new_prompt = typed

# --- TRAITEMENT ---
if new_prompt:
    st.session_state.messages.append({"role": "user", "content": new_prompt})
    with st.chat_message("user"):
        st.markdown(new_prompt)
    
    answer, sources = None, []
    with st.chat_message("assistant"):
        if st.session_state.rag is None:
            with st.spinner("Initialisation du pipeline..."):
                rag, error = load_rag()
            if error:
                st.error(f"Erreur : {error}")
                st.stop()
            st.session_state.rag = rag
        
        with st.spinner("Recherche dans les rapports scientifiques..."):
            try:
                result = st.session_state.rag.ask(new_prompt)
                answer = result["answer"]
                sources = result.get("sources", [])
            except Exception as e:
                st.error(f"Erreur : {e}")
        
        if answer:
            st.markdown(answer)
            if sources:
                with st.expander(f"Sources utilisees ({len(sources)})"):
                    for src in sources:
                        st.markdown(f"""
<div class="cr-source">
    <div class="cr-source-name">{src['source']} - page {src['page']}</div>
    <div class="cr-source-meta">FAISS {src['faiss_score']:.3f} - Rerank {src['rerank_score']:.3f}</div>
</div>
""", unsafe_allow_html=True)
    
    if answer:
        st.session_state.messages.append({"role": "assistant", "content": answer, "sources": sources})
        st.session_state.did_you_know_idx = random.randint(0, len(FACTS) - 1)

# --- FOOTER ---
st.markdown("""
<div class="cr-footer">
    ClimateRAG - Reponses fondees sur les rapports du GIEC (AR6), Copernicus et Meteo-France<br>
    Les reponses sont generees automatiquement - verifiez toujours les sources originales.<br>
    Fait avec conviction : comprendre le climat, c'est deja agir.
</div>
""", unsafe_allow_html=True)