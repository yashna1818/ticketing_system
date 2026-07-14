import streamlit as st
import os
import tempfile
import time
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from speech_to_text import transcribe_audio
from text_to_speech import synthesize_speech
from preprocessing import get_sentiment_and_priority
from model import TicketClassifier
from db_helper import add_ticket, get_all_tickets, update_ticket, delete_ticket, clear_all_tickets
from translator import translate_text, detect_language, transliterate_text
from priority_classifier import PriorityClassifier

# Page Config must be the first Streamlit command
st.set_page_config(
    page_title="AuraSupport AI Console", 
    page_icon="🎧", 
    layout="wide", 
    initial_sidebar_state="expanded"
)

# Initialize Session State for Navigation
if "app_mode" not in st.session_state:
    st.session_state.app_mode = "🏢 Departments Overview"

# Language Selector in Sidebar
st.sidebar.markdown("""
<div style="text-align: center; padding: 5px 0;">
    <span style="font-size: 2.2rem;">🌐</span>
</div>
""", unsafe_allow_html=True)
sys_lang = st.sidebar.selectbox(
    "System Language / ಭಾಷೆ / भाषा:", 
    ["English", "ಕನ್ನಡ (Kannada)", "हिंदी (Hindi)"], 
    index=0
)
lang_map = {
    "English": "en",
    "ಕನ್ನಡ (Kannada)": "kn",
    "हिंदी (Hindi)": "hi"
}
target_lang = lang_map[sys_lang]

@st.cache_data
def get_cached_translation(text, dest_lang):
    return translate_text(text, source_lang='en', target_lang=dest_lang)

def T(text):
    if not text:
        return ""
    if target_lang == 'en':
        return text
    return get_cached_translation(text, target_lang)

# Inject Custom Global CSS for a Premium, Interactive Glassmorphic Cyberpunk Theme
# Note: Broad [class*="css"] is removed to prevent layout and text overlapping glitches.
st.markdown("""
<style>
    @import url('https://fonts.googleapis.com/css2?family=Outfit:wght@300;400;500;600;700;800&display=swap');
    
    /* Global font overrides */
    html, body, .stMarkdown, p, td, th {
        font-family: 'Outfit', sans-serif;
        color: #e2e8f0;
    }
    
    /* Main Background Overrides */
    .stApp {
        background: radial-gradient(circle at 50% 50%, #121620 0%, #0a0d12 100%);
    }
    
    /* Gradient Main Title */
    .main-title {
        background: linear-gradient(135deg, #a29bfe 0%, #6c5ce7 30%, #ff7675 70%, #ffeaa7 100%);
        background-size: 200% auto;
        -webkit-background-clip: text;
        -webkit-text-fill-color: transparent;
        font-weight: 800;
        font-size: 3.5rem;
        padding-bottom: 5px;
        margin-bottom: -5px;
        text-align: center;
        animation: gradient-flow 8s ease infinite;
        letter-spacing: -1px;
    }
    
    @keyframes gradient-flow {
        0% { background-position: 0% 50%; }
        50% { background-position: 100% 50%; }
        100% { background-position: 0% 50%; }
    }
    
    .sub-title {
        text-align: center;
        color: #94a3b8;
        font-size: 1.25rem;
        margin-bottom: 2rem;
        font-weight: 300;
        letter-spacing: 0.5px;
        max-width: 800px;
        margin-left: auto;
        margin-right: auto;
    }
    
    /* Custom separator line */
    .grad-divider {
        height: 2px; 
        background: linear-gradient(90deg, transparent, rgba(108, 99, 255, 0.4), rgba(255, 101, 132, 0.4), transparent); 
        margin: 25px auto;
        width: 80%;
    }
    
    /* Sleek Glassmorphic Card Container */
    .glass-card {
        background: rgba(30, 41, 59, 0.4);
        border: 1px solid rgba(255, 255, 255, 0.08);
        border-radius: 20px;
        padding: 24px;
        margin-top: 10px;
        margin-bottom: 20px;
        box-shadow: 0 15px 35px 0 rgba(0, 0, 0, 0.4);
        backdrop-filter: blur(20px);
        -webkit-backdrop-filter: blur(20px);
        transition: all 0.4s cubic-bezier(0.165, 0.84, 0.44, 1);
    }
    
    .glass-card:hover {
        border-color: rgba(108, 99, 255, 0.35);
        box-shadow: 0 25px 50px 0 rgba(108, 99, 255, 0.12);
        transform: translateY(-3px);
    }

    /* Premium Badges */
    .badge {
        padding: 5px 12px;
        border-radius: 10px;
        font-weight: 600;
        font-size: 0.8rem;
        display: inline-block;
        letter-spacing: 0.5px;
        text-transform: uppercase;
    }
    
    .badge-High { background-color: rgba(239, 68, 68, 0.15); color: #f87171; border: 1px solid rgba(239, 68, 68, 0.3); }
    .badge-Medium { background-color: rgba(245, 158, 11, 0.15); color: #fbbf24; border: 1px solid rgba(245, 158, 11, 0.3); }
    .badge-Low { background-color: rgba(16, 185, 129, 0.15); color: #34d399; border: 1px solid rgba(16, 185, 129, 0.3); }
    
    .badge-Negative { background-color: rgba(239, 68, 68, 0.15); color: #f87171; border: 1px solid rgba(239, 68, 68, 0.3); }
    .badge-Neutral { background-color: rgba(148, 163, 184, 0.15); color: #cbd5e1; border: 1px solid rgba(148, 163, 184, 0.3); }
    .badge-Positive { background-color: rgba(16, 185, 129, 0.15); color: #34d399; border: 1px solid rgba(16, 185, 129, 0.3); }
    
    .badge-New { background-color: rgba(99, 102, 241, 0.15); color: #818cf8; border: 1px solid rgba(99, 102, 241, 0.3); }
    .badge-In-Progress { background-color: rgba(245, 158, 11, 0.15); color: #fbbf24; border: 1px solid rgba(245, 158, 11, 0.3); }
    .badge-Resolved { background-color: rgba(16, 185, 129, 0.15); color: #34d399; border: 1px solid rgba(16, 185, 129, 0.3); }
    
    .badge-Category { background-color: rgba(139, 92, 246, 0.12); color: #a78bfa; border: 1px solid rgba(139, 92, 246, 0.25); }
    
    /* Custom forms & text inputs */
    div[data-baseweb="textarea"], div[data-baseweb="input"] {
        background-color: rgba(15, 23, 42, 0.55) !important;
        border: 1px solid rgba(255, 255, 255, 0.1) !important;
        border-radius: 12px !important;
        transition: all 0.3s ease;
    }
    
    /* Styled buttons */
    .stButton > button {
        background: linear-gradient(135deg, #6366f1 0%, #a855f7 100%) !important;
        color: white !important;
        border: none !important;
        border-radius: 12px !important;
        padding: 8px 18px !important;
        font-weight: 600 !important;
        font-size: 0.9rem !important;
        letter-spacing: 0.5px !important;
        box-shadow: 0 6px 15px rgba(99, 102, 241, 0.25) !important;
        transition: all 0.3s cubic-bezier(0.175, 0.885, 0.32, 1.275) !important;
        width: 100% !important;
    }
    
    .stButton > button:hover {
        transform: translateY(-2px) !important;
        box-shadow: 0 10px 25px rgba(168, 85, 247, 0.4) !important;
        background: linear-gradient(135deg, #a855f7 0%, #6366f1 100%) !important;
    }
    
    .stButton > button:active {
        transform: translateY(1px) !important;
    }
    
    /* Sidebar Navigation Style */
    section[data-testid="stSidebar"] {
        background: linear-gradient(180deg, #080c14 0%, #040609 100%) !important;
        border-right: 1px solid rgba(255, 255, 255, 0.04) !important;
    }

    /* Floating effect for Hero Cards */
    .hero-stat-card {
        background: rgba(30, 41, 59, 0.25);
        border: 1px solid rgba(255, 255, 255, 0.05);
        border-radius: 16px;
        padding: 20px;
        text-align: center;
        box-shadow: 0 8px 25px rgba(0, 0, 0, 0.3);
        backdrop-filter: blur(10px);
        transition: all 0.3s ease;
    }
    .hero-stat-card:hover {
        transform: translateY(-4px);
        border-color: rgba(99, 102, 241, 0.25);
        box-shadow: 0 12px 30px rgba(99, 102, 241, 0.1);
    }
    
    /* Style Streamlit's native container borders to look like glass cards */
    div[data-testid="stVerticalBlockBorder"] {
        background: rgba(30, 41, 59, 0.4) !important;
        border: 1px solid rgba(255, 255, 255, 0.08) !important;
        border-radius: 20px !important;
        padding: 24px !important;
        box-shadow: 0 15px 35px 0 rgba(0, 0, 0, 0.4) !important;
        backdrop-filter: blur(20px) !important;
        -webkit-backdrop-filter: blur(20px) !important;
        transition: all 0.4s cubic-bezier(0.165, 0.84, 0.44, 1) !important;
    }
    
    div[data-testid="stVerticalBlockBorder"]:hover {
        border-color: rgba(99, 102, 241, 0.35) !important;
        box-shadow: 0 25px 50px 0 rgba(99, 102, 241, 0.12) !important;
        transform: translateY(-3px) !important;
    }
</style>
""", unsafe_allow_html=True)

# Navigation Sidebar
st.sidebar.title(T("🧭 Navigation"))
modes = ["🏢 Departments Overview", "🎙️ AI Ticket Analyzer", "📞 Voice Call Agent", "📊 Model Benchmarking", "📥 Admin Ticket Queue"]
current_index = modes.index(st.session_state.app_mode) if st.session_state.app_mode in modes else 0

selected_mode = st.sidebar.radio(
    T("Go to:"), 
    modes,
    index=current_index,
    format_func=T
)

if selected_mode != st.session_state.app_mode:
    st.session_state.app_mode = selected_mode
    st.rerun()

app_mode = st.session_state.app_mode
st.sidebar.markdown("---")

# Model Loading & Caching
@st.cache_resource(show_spinner="Loading ML Classifiers...")
def load_classifier():
    clf = TicketClassifier()
    loaded = clf.load()
    return clf, loaded

@st.cache_resource(show_spinner="Loading Priority Classifier...")
def load_priority_classifier():
    pclf = PriorityClassifier()
    loaded = pclf.load()
    if not loaded:
        pclf.train()
        pclf.save()
    return pclf

clf, is_loaded = load_classifier()
priority_clf = load_priority_classifier()

# Active Model Selector
if is_loaded:
    st.sidebar.subheader("🤖 Active Classifier Model")
    selected_model_key = st.sidebar.selectbox(
        "Choose Active Model:",
        ["Logistic Regression", "Naive Bayes", "Support Vector Machine (SVC)"],
        index=0
    )
    model_key_map = {
        "Logistic Regression": "logistic",
        "Naive Bayes": "naive_bayes",
        "Support Vector Machine (SVC)": "svc"
    }
    active_model = model_key_map[selected_model_key]
else:
    active_model = 'logistic'

# Priority classifier sidebar status
if priority_clf.is_trained:
    acc = priority_clf.train_metrics.get('accuracy', 0)
    st.sidebar.success(f"🎯 Priority ML: Active ({acc:.0%} acc)")
else:
    st.sidebar.warning("⚠️ Priority ML: Rule-based fallback")

# Whisper Configuration Sidebar
st.sidebar.subheader("🎙️ Whisper STT Config")
whisper_model_size = st.sidebar.selectbox(
    "Local Model Size:",
    ["tiny", "base", "small"],
    index=1,
    help="Smaller models run faster. Larger models are more accurate."
)

openai_api_key = None  # Cloud Whisper disabled; using local Whisper only

# Common Rendering Card for Analysis
def render_analysis_card(transcribed_text, active_model='svc', ticket_id=None):
    all_preds = clf.predict_all(transcribed_text)
    category = all_preds[active_model]
    sentiment, _ = get_sentiment_and_priority(transcribed_text)  # sentiment only
    priority, explanation, confidence = priority_clf.predict_with_explanation(transcribed_text, category)
    
    dept_mapping = {
        "Billing Issue": "Finance & Billing",
        "Refund Request": "Finance & Billing",
        "Technical Issue": "IT & Engineering",
        "Account Access": "Security & Auth",
        "General Inquiry": "Tier 1 Support"
    }
    routed_dept = dept_mapping.get(category, "Tier 1 Support")
    
    # Confidence bar widths (percentage strings)
    conf_h  = confidence.get('High',   0.0)
    conf_m  = confidence.get('Medium', 0.0)
    conf_l  = confidence.get('Low',    0.0)
    
    res_col1, res_col2 = st.columns([1.5, 1])
    with res_col1:
        st.markdown(f"""
        <div class="glass-card">
            <h4 style="margin-top:0px; color: #f8f9fa;">
                🗣️ {T("Official Call Transcript")} {f' (Ticket #{ticket_id})' if ticket_id else ''}
            </h4>
            <p style="font-size: 1.25rem; color: #f1f5f9; line-height: 1.6; font-style:italic; border-left: 4px solid #6366f1; padding-left: 20px; margin-top: 20px;">
                "{transcribed_text}"
            </p>
        </div>
        """, unsafe_allow_html=True)
        
        st.markdown(f"""
        <div class="glass-card">
            <h4 style="margin-top:0px; color: #f8f9fa;">⚖️ {T("Model Prediction Consensus")}</h4>
            <table style="width:100%; border-collapse: collapse; margin-top: 15px; color: #cbd5e1;">
                <tr style="border-bottom: 2px solid rgba(255,255,255,0.08); font-weight: 600;">
                    <th style="text-align: left; padding: 10px 8px; color: #94a3b8; font-size: 0.95rem; text-transform: uppercase;">Model</th>
                    <th style="text-align: left; padding: 10px 8px; color: #94a3b8; font-size: 0.95rem; text-transform: uppercase;">Prediction</th>
                </tr>
                <tr style="border-bottom: 1px solid rgba(255,255,255,0.04);">
                    <td style="padding: 12px 8px;">Logistic Regression</td>
                    <td style="padding: 12px 8px;"><span class="badge badge-Category">{all_preds['logistic']}</span></td>
                </tr>
                <tr style="border-bottom: 1px solid rgba(255,255,255,0.04);">
                    <td style="padding: 12px 8px;">Naive Bayes</td>
                    <td style="padding: 12px 8px;"><span class="badge badge-Category">{all_preds['naive_bayes']}</span></td>
                </tr>
                <tr>
                    <td style="padding: 12px 8px;">Support Vector Machine (SVC)</td>
                    <td style="padding: 12px 8px;"><span class="badge badge-Category">{all_preds['svc']}</span></td>
                </tr>
            </table>
        </div>
        """, unsafe_allow_html=True)
        
    with res_col2:
        # Priority confidence mini-bars
        bar_h = f"{conf_h * 100:.0f}%"
        bar_m = f"{conf_m * 100:.0f}%"
        bar_l = f"{conf_l * 100:.0f}%"
        
        st.markdown(f"""
        <div class="glass-card" style="background: linear-gradient(135deg, rgba(30, 41, 59, 0.7), rgba(99, 102, 241, 0.05)); border-color: rgba(99, 102, 241, 0.2);">
            <h4 style="margin-top:0px; color: #f8f9fa;">🎛️ {T("AI Routing Analysis")} ({active_model.upper()})</h4>
            <br>
            <div style="border-bottom: 1px solid rgba(255,255,255,0.05); padding-bottom: 12px;">
                <span style="color: #94a3b8; font-size: 0.9rem;">Routed Department</span><br>
                <span class="badge" style="background-color:rgba(168, 85, 247, 0.2); color:#c084fc; border:1px solid #a855f7; margin-top: 5px;">{routed_dept}</span>
            </div>
            <br>
            <div style="border-bottom: 1px solid rgba(255,255,255,0.05); padding-bottom: 12px;">
                <span style="color: #94a3b8; font-size: 0.9rem;">AI Classification</span><br>
                <span class="badge badge-Category" style="margin-top: 5px;">{category}</span>
            </div>
            <br>
            <div style="border-bottom: 1px solid rgba(255,255,255,0.05); padding-bottom: 12px;">
                <span style="color: #94a3b8; font-size: 0.9rem;">Customer Tone</span><br>
                <span class="badge badge-{sentiment}" style="margin-top: 5px;">{sentiment}</span>
            </div>
            <br>
            <div style="border-bottom: 1px solid rgba(255,255,255,0.05); padding-bottom: 14px;">
                <span style="color: #94a3b8; font-size: 0.9rem;">Urgency Level</span><br>
                <span class="badge badge-{priority}" style="margin-top: 5px;">{priority} Priority</span>
            </div>
            <br>
            <div>
                <span style="color: #94a3b8; font-size: 0.85rem; font-weight: 600; text-transform: uppercase; letter-spacing: 0.5px;">🎯 Priority ML Confidence</span>
                <div style="margin-top: 10px; display: flex; flex-direction: column; gap: 6px;">
                    <div style="display: flex; align-items: center; gap: 8px;">
                        <span style="color: #f87171; font-size: 0.8rem; width: 48px;">High</span>
                        <div style="flex: 1; background: rgba(255,255,255,0.06); border-radius: 4px; height: 8px; overflow: hidden;">
                            <div style="width: {bar_h}; height: 100%; background: linear-gradient(90deg, #ef4444, #f87171); border-radius: 4px; transition: width 0.5s;"></div>
                        </div>
                        <span style="color: #94a3b8; font-size: 0.75rem; width: 34px; text-align: right;">{bar_h}</span>
                    </div>
                    <div style="display: flex; align-items: center; gap: 8px;">
                        <span style="color: #fbbf24; font-size: 0.8rem; width: 48px;">Med</span>
                        <div style="flex: 1; background: rgba(255,255,255,0.06); border-radius: 4px; height: 8px; overflow: hidden;">
                            <div style="width: {bar_m}; height: 100%; background: linear-gradient(90deg, #d97706, #fbbf24); border-radius: 4px; transition: width 0.5s;"></div>
                        </div>
                        <span style="color: #94a3b8; font-size: 0.75rem; width: 34px; text-align: right;">{bar_m}</span>
                    </div>
                    <div style="display: flex; align-items: center; gap: 8px;">
                        <span style="color: #34d399; font-size: 0.8rem; width: 48px;">Low</span>
                        <div style="flex: 1; background: rgba(255,255,255,0.06); border-radius: 4px; height: 8px; overflow: hidden;">
                            <div style="width: {bar_l}; height: 100%; background: linear-gradient(90deg, #059669, #34d399); border-radius: 4px; transition: width 0.5s;"></div>
                        </div>
                        <span style="color: #94a3b8; font-size: 0.75rem; width: 34px; text-align: right;">{bar_l}</span>
                    </div>
                </div>
                <div style="margin-top: 12px; padding: 8px 10px; background: rgba(99,102,241,0.08); border-radius: 8px; border-left: 3px solid rgba(99,102,241,0.5);">
                    <span style="color: #a5b4fc; font-size: 0.78rem; line-height: 1.5;">{explanation}</span>
                </div>
            </div>
        </div>
        """, unsafe_allow_html=True)


# --- 1. Departments Overview (Landing Page) ---
if app_mode == "🏢 Departments Overview":
    st.markdown(f"### 🏢 {T('Customer Support Ticket')}", unsafe_allow_html=True)
    
    # Real-time Stats Header
    tickets = get_all_tickets()
    tot = len(tickets)
    high = len([t for t in tickets if t['priority'] == 'High'])
    open_t = len([t for t in tickets if t['status'] in ['New', 'In Progress']])
    res_t = len([t for t in tickets if t['status'] == 'Resolved'])
    
    stat_col1, stat_col2, stat_col3, stat_col4 = st.columns(4)
    with stat_col1:
        st.markdown(f"""
        <div class="hero-stat-card">
            <span style="font-size: 2.8rem; font-weight: 800; color: #818cf8; display: block; margin-bottom: 2px;">{tot}</span>
            <span style="font-size: 0.85rem; color: #94a3b8; font-weight: 600; text-transform: uppercase; letter-spacing: 1px;">{T("Total Tickets")}</span>
        </div>
        """, unsafe_allow_html=True)
    with stat_col2:
        st.markdown(f"""
        <div class="hero-stat-card">
            <span style="font-size: 2.8rem; font-weight: 800; color: #f87171; display: block; margin-bottom: 2px;">{high}</span>
            <span style="font-size: 0.85rem; color: #94a3b8; font-weight: 600; text-transform: uppercase; letter-spacing: 1px;">🚨 {T("High Priority")}</span>
        </div>
        """, unsafe_allow_html=True)
    with stat_col3:
        st.markdown(f"""
        <div class="hero-stat-card">
            <span style="font-size: 2.8rem; font-weight: 800; color: #fbbf24; display: block; margin-bottom: 2px;">{open_t}</span>
            <span style="font-size: 0.85rem; color: #94a3b8; font-weight: 600; text-transform: uppercase; letter-spacing: 1px;">📁 {T("Open Tickets")}</span>
        </div>
        """, unsafe_allow_html=True)
    with stat_col4:
        st.markdown(f"""
        <div class="hero-stat-card">
            <span style="font-size: 2.8rem; font-weight: 800; color: #34d399; display: block; margin-bottom: 2px;">{res_t}</span>
            <span style="font-size: 0.85rem; color: #94a3b8; font-weight: 600; text-transform: uppercase; letter-spacing: 1px;">✅ {T("Resolved")}</span>
        </div>
        """, unsafe_allow_html=True)

    st.markdown("<br>", unsafe_allow_html=True)

    # Core Features Landing Grid
    col_feat1, col_feat2 = st.columns(2)
    with col_feat1:
        with st.container(border=True):
            st.markdown(f"""
            <div style="font-size: 2.5rem; margin-bottom: 5px;">🎙️</div>
            <h3 style="color: #818cf8; margin-top: 5px; font-weight: 700;">{T("Voice Analyzer & Transcriber")}</h3>
            <p style="color: #cbd5e1; font-size: 0.95rem; line-height: 1.5; min-height: 70px;">
                {T("Record voice queries or upload audio. Whisper STT transcribes, while our classifiers auto-route, assess sentiment, and prioritize.")}
            </p>
            """, unsafe_allow_html=True)
            if st.button("🚀 " + T("Launch AI Analyzer"), key="btn_go_analyzer", width='stretch'):
                st.session_state.app_mode = "🎙️ AI Ticket Analyzer"
                st.rerun()
                
        st.markdown("<br>", unsafe_allow_html=True)
        
        with st.container(border=True):
            st.markdown(f"""
            <div style="font-size: 2.5rem; margin-bottom: 5px;">📞</div>
            <h3 style="color: #34d399; margin-top: 5px; font-weight: 700;">{T("Real-Time Simulated Phone Call")}</h3>
            <p style="color: #cbd5e1; font-size: 0.95rem; line-height: 1.5; min-height: 70px;">
                {T("Experience a two-way voice call support agent. Speech synthesis answers back in selected system languages dynamically.")}
            </p>
            """, unsafe_allow_html=True)
            if st.button("📞 " + T("Start Voice Call Sim"), key="btn_go_voice", width='stretch'):
                st.session_state.app_mode = "📞 Voice Call Agent"
                st.rerun()
                
    with col_feat2:
        with st.container(border=True):
            st.markdown(f"""
            <div style="font-size: 2.5rem; margin-bottom: 5px;">📊</div>
            <h3 style="color: #c084fc; margin-top: 5px; font-weight: 700;">{T("Multi-Model ML Benchmarking")}</h3>
            <p style="color: #cbd5e1; font-size: 0.95rem; line-height: 1.5; min-height: 70px;">
                {T("Analyze classifiers (Logistic Regression, Naive Bayes, SVM). Review validation metrics and retrain models on the fly.")}
            </p>
            """, unsafe_allow_html=True)
            if st.button("📊 " + T("Check Model Diagnostics"), key="btn_go_bench", width='stretch'):
                st.session_state.app_mode = "📊 Model Benchmarking"
                st.rerun()
                
        st.markdown("<br>", unsafe_allow_html=True)
        
        with st.container(border=True):
            st.markdown(f"""
            <div style="font-size: 2.5rem; margin-bottom: 5px;">📥</div>
            <h3 style="color: #fbbf24; margin-top: 5px; font-weight: 700;">{T("Live Admin Queue Console")}</h3>
            <p style="color: #cbd5e1; font-size: 0.95rem; line-height: 1.5; min-height: 70px;">
                {T("Inspect details in the SQLite dispatch queue. Manually adjust ticket statuses, save notes, and speak responses.")}
            </p>
            """, unsafe_allow_html=True)
            if st.button("📥 " + T("Open Queue Manager"), key="btn_go_queue", width='stretch'):
                st.session_state.app_mode = "📥 Admin Ticket Queue"
                st.rerun()

    st.markdown("<br>", unsafe_allow_html=True)
    
    # Department Map / Architecture section
    st.markdown(f"### 🏦 {T('Intelligent Routing Architecture Mapping')}")
    col_dept1, col_dept2, col_dept3, col_dept4 = st.columns(4)
    with col_dept1:
        st.markdown(f"""
        <div class="glass-card" style="text-align: center; border-color: rgba(99, 102, 241, 0.15); height: 160px;">
            <div style="font-size: 1.8rem;">🧑‍💻</div>
            <h4 style="color: #818cf8; margin-top: 5px; font-size: 1rem;">{T("IT & Engineering")}</h4>
            <p style="font-size: 0.82rem; color: #94a3b8;">Bugs, system down alerts, errors, crashes.</p>
        </div>
        """, unsafe_allow_html=True)
    with col_dept2:
        st.markdown(f"""
        <div class="glass-card" style="text-align: center; border-color: rgba(168, 85, 247, 0.15); height: 160px;">
            <div style="font-size: 1.8rem;">💳</div>
            <h4 style="color: #c084fc; margin-top: 5px; font-size: 1rem;">{T("Finance & Billing")}</h4>
            <p style="font-size: 0.82rem; color: #94a3b8;">Subscription fees, invoice claims, refunds.</p>
        </div>
        """, unsafe_allow_html=True)
    with col_dept3:
        st.markdown(f"""
        <div class="glass-card" style="text-align: center; border-color: rgba(16, 185, 129, 0.15); height: 160px;">
            <div style="font-size: 1.8rem;">🔐</div>
            <h4 style="color: #34d399; margin-top: 5px; font-size: 1rem;">{T("Security & Auth")}</h4>
            <p style="font-size: 0.82rem; color: #94a3b8;">Authentication locks, security breaches.</p>
        </div>
        """, unsafe_allow_html=True)
    with col_dept4:
        st.markdown(f"""
        <div class="glass-card" style="text-align: center; border-color: rgba(245, 158, 11, 0.15); height: 160px;">
            <div style="font-size: 1.8rem;">⚡</div>
            <h4 style="color: #fbbf24; margin-top: 5px; font-size: 1rem;">{T("Priority Escalation")}</h4>
            <p style="font-size: 0.82rem; color: #94a3b8;">Auto-detecting fraud, legal issues, or anger.</p>
        </div>
        """, unsafe_allow_html=True)


# --- 2. AI Ticket Analyzer ---
elif app_mode == "🎙️ AI Ticket Analyzer":
    if not is_loaded:
        st.error("🚨 ML Models Missing! Train model first.")
        st.stop()

    st.sidebar.header("🕹️ Ticket Input Controls")
    input_method = st.sidebar.radio("Select Input Flow", ["🎙️ Single Voice Note", "📁 Upload File", "⌨️ Text Editor"])
    
    audio_path = None
    text_input = ""

    col_input1, col_input2, col_input3 = st.columns([1, 6, 1])
    with col_input2:
        with st.container(border=True):
            st.write("#### 📝 Customer Query Input")
            if input_method == "🎙️ Single Voice Note":
                audio_value = st.audio_input("Record Audio", label_visibility="hidden")
                if audio_value:
                    with tempfile.NamedTemporaryFile(delete=False, suffix=".wav") as f:
                        f.write(audio_value.getbuffer())
                        audio_path = f.name
            elif input_method == "📁 Upload File":
                uploaded_file = st.file_uploader("Upload customer audio (.wav, .mp3)", type=["wav", "mp3", "m4a", "ogg"])
                if uploaded_file is not None:
                    with tempfile.NamedTemporaryFile(delete=False, suffix=".wav") as f:
                        f.write(uploaded_file.getbuffer())
                        audio_path = f.name
            else:
                text_input = st.text_area("✍️ Customer Message Context", height=150, placeholder="Type customer query here... e.g. I cannot access my account because my password reset is broken.")

            analyze_btn = st.button("🚀 Analyze Support Ticket", width='stretch', type="primary")

    if analyze_btn:
        if not audio_path and not text_input.strip():
            st.warning("⚠️ Please provide an audio recording, file, or text input first.")
        else:
            with st.spinner("🤖 Whisper is transcribing & ML is analyzing..."):
                if audio_path:
                    transcribed_text = transcribe_audio(audio_path, model_name=whisper_model_size, api_key=openai_api_key)
                    try:
                        os.remove(audio_path)
                    except Exception:
                        pass
                else:
                    transcribed_text = text_input

                if transcribed_text.strip():
                    # Calculate ML classification
                    all_preds = clf.predict_all(transcribed_text)
                    category = all_preds[active_model]
                    sentiment, _ = get_sentiment_and_priority(transcribed_text)
                    priority = priority_clf.predict(transcribed_text, category)
                    
                    # Save to DB
                    ticket_id = add_ticket(
                        transcript=transcribed_text,
                        category=category,
                        sentiment=sentiment,
                        priority=priority,
                        model_used=active_model
                    )
                    
                    st.success(f"🎟️ Ticket #{ticket_id} created and queued successfully!")
                    render_analysis_card(transcribed_text, active_model, ticket_id=ticket_id)
                else:
                    st.error("Input was empty or unintelligible.")


# --- 3. Voice Call Agent ---
elif app_mode == "📞 Voice Call Agent":
    if not is_loaded:
        st.error("🚨 ML Models Missing! Train model first.")
        st.stop()
        
    # Reset conversation if target language changes
    if "agent_lang" in st.session_state and st.session_state.agent_lang != target_lang:
        for key in ["messages", "chat_finished", "pending_audio", "last_processed_audio_hash", "agent_lang"]:
            if key in st.session_state:
                del st.session_state[key]

    st.markdown(f"### 📞 {T('Real-Time Live Phone Call Support Simulation')}")
    st.markdown(T("Experience a two-way AI voice call! Speak to the Agent, and the Agent will reply back. Once the call is resolved, the transcript is analyzed and queued in the DB."))
    
    if "messages" not in st.session_state:
        st.session_state.messages = []
        st.session_state.chat_finished = False
        greeting = T("Hello! Welcome to our Support Desk. I am your AI agent. Please speak clearly using your microphone, and tell me: How can I assist you today?")
        st.session_state.messages.append({"role": "assistant", "content": greeting})
        st.session_state.pending_audio = synthesize_speech(greeting, lang=target_lang)
        st.session_state.agent_lang = target_lang

    # Initialize audio cache tracker
    if "last_processed_audio_hash" not in st.session_state:
        st.session_state.last_processed_audio_hash = None
    if "audio_input_counter" not in st.session_state:
        st.session_state.audio_input_counter = 0

    # Render Visual Conversation History
    for msg in st.session_state.messages:
        avatar = "🤖" if msg["role"] == "assistant" else "👤"
        with st.chat_message(msg["role"], avatar=avatar):
            st.write(msg["content"])
            if msg["role"] == "assistant" and msg.get("transliterated"):
                st.caption(f"✍️ *Transliteration (Latin):* {msg['transliterated']}")
            
    # Autoplay pending AI voice if exists
    if "pending_audio" in st.session_state and st.session_state.pending_audio is not None:
        st.audio(st.session_state.pending_audio, format="audio/mp3", autoplay=True)
        st.session_state.pending_audio = None
            
    if not st.session_state.chat_finished:
        st.markdown("---")
        # Voice Input Component
        audio_value = st.audio_input(
            T("🗣️ It's your turn to speak..."), 
            key=f"voice_agent_audio_{st.session_state.audio_input_counter}"
        )
        
        if audio_value:
            audio_bytes = audio_value.getbuffer().tobytes()
            audio_hash = hash(audio_bytes)
            
            if st.session_state.last_processed_audio_hash != audio_hash:
                st.session_state.last_processed_audio_hash = audio_hash
                
                with st.spinner("Agent is listening & transcribing..."):
                    with tempfile.NamedTemporaryFile(delete=False, suffix=".wav") as f:
                        f.write(audio_bytes)
                        audio_path = f.name
                        
                    user_text = transcribe_audio(audio_path, model_name=whisper_model_size, api_key=openai_api_key, language=target_lang)
                    os.remove(audio_path)
                    
                    st.session_state.messages.append({"role": "user", "content": user_text})
                    st.session_state.audio_input_counter += 1
                    st.rerun()
            else:
                pass
                
        # Generate Agent Reply
        if len(st.session_state.messages) > 0 and st.session_state.messages[-1]["role"] == "user":
            with st.spinner("Agent is thinking and generating a voice reply..."):
                time.sleep(1)
                user_msgs = [m for m in st.session_state.messages if m["role"] == "user"]
                last_user_text = user_msgs[-1]["content"]
                
                detected_lang = target_lang
                st.session_state.agent_lang = target_lang
                
                last_user_text_en = translate_text(last_user_text, source_lang=detected_lang, target_lang='en').lower()
                
                if len(user_msgs) == 1:
                    if any(w in last_user_text_en for w in ['log in', 'login', 'password', 'account', 'auth', 'access', 'blocked', 'locked', 'reset']):
                        reply_en = "I understand you are having account access issues. Could you please specify if you are seeing any specific error code, and let me know if you tried resetting it?"
                    elif any(w in last_user_text_en for w in ['bill', 'payment', 'charge', 'invoice', 'fee', 'deduct', 'money', 'card', 'visa', 'mastercard']):
                        reply_en = "I see this is regarding a billing or payment concern. Could you confirm the last four digits of your card, or if you have an invoice number?"
                    elif any(w in last_user_text_en for w in ['refund', 'return', 'cancel', 'subscription']):
                        reply_en = "Got it, a refund or subscription query. Could you let me know the purchase date and whether the product has been cancelled?"
                    elif any(w in last_user_text_en for w in ['bug', 'crash', 'error', 'sync', 'load', 'update', 'performance', 'tech', 'software', 'fail', 'slow', 'broken']):
                        reply_en = "It sounds like a technical issue. Could you tell me which operating system or browser you are using when this happens?"
                    else:
                        reply_en = "I've noted that down. Could you provide a bit more detail about the exact issue so I can route it to the right team?"
                else:
                    reply_en = "Thank you so much for securely providing those details! Our call is now concluded. I will immediately escalate this transcript to the optimal department. Have a nice day!"
                    st.session_state.chat_finished = True
                
                reply = translate_text(reply_en, source_lang='en', target_lang=detected_lang)
                
                transliterated = ""
                if detected_lang in ['kn', 'hi']:
                    transliterated = transliterate_text(reply, detected_lang)
                    
                st.session_state.messages.append({
                    "role": "assistant", 
                    "content": reply,
                    "transliterated": transliterated
                })
                st.session_state.pending_audio = synthesize_speech(reply, lang=detected_lang)
                st.rerun()
 
    else:
        st.success("☎️ Call Dropped. Conversation Archived. Generating Live ML Routing Trace...")
        
        full_transcript = " ".join([m["content"] for m in st.session_state.messages if m["role"] == "user"])
        detected_lang = st.session_state.get("agent_lang", target_lang)
        
        if detected_lang in ['kn', 'hi']:
            full_transcript_en = translate_text(full_transcript, source_lang=detected_lang, target_lang='en')
        else:
            full_transcript_en = full_transcript
            
        all_preds = clf.predict_all(full_transcript_en)
        category = all_preds[active_model]
        sentiment, _ = get_sentiment_and_priority(full_transcript_en)
        priority = priority_clf.predict(full_transcript_en, category)
        
        ticket_id = add_ticket(
            transcript=full_transcript,
            category=category,
            sentiment=sentiment,
            priority=priority,
            model_used=active_model,
            language=detected_lang,
            translation=full_transcript_en if detected_lang in ['kn', 'hi'] else ""
        )
        
        st.success(f"🎟️ Ticket #{ticket_id} created and queued successfully!")
        render_analysis_card(full_transcript, active_model, ticket_id=ticket_id)
        
        if st.button("🔄 Initiate New Call", width='stretch'):
            if "messages" in st.session_state:
                del st.session_state.messages
            if "chat_finished" in st.session_state:
                del st.session_state.chat_finished
            st.session_state.last_processed_audio_hash = None
            st.rerun()


# --- 4. Model Benchmarking ---
elif app_mode == "📊 Model Benchmarking":
    st.markdown("### 📊 Multi-Model Performance Comparison")
    st.markdown("Compare the performance of our 3 trained models: **Logistic Regression**, **Naive Bayes (MultinomialNB)**, and **Support Vector Machine (LinearSVC)**.")
    
    if not is_loaded or not clf.metrics:
        st.warning("⚠️ Benchmarking metrics are not available. Please run model training below to generate metrics.")
    else:
        metrics_df = pd.DataFrame(clf.metrics).T
        metrics_df = metrics_df.round(4)
        
        col_m1, col_m2 = st.columns([1.2, 1])
        with col_m1:
            with st.container(border=True):
                st.subheader("📈 Performance Metrics Table")
                st.table(metrics_df)
        
        with col_m2:
            with st.container(border=True):
                st.subheader("📊 F1-Score & Accuracy Comparison")
                chart_data = metrics_df[['Accuracy', 'F1-score']]
                st.bar_chart(chart_data)
            
        with st.container(border=True):
            st.subheader("🧩 Advanced Model Diagnostic Visualizations")
        
        bench_model_ui = st.selectbox(
            "Select Model to Inspect:",
            ["Logistic Regression", "Naive Bayes", "Support Vector Machine (SVC)"]
        )
        model_ui_map = {
            "Logistic Regression": "logistic",
            "Naive Bayes": "naive_bayes",
            "Support Vector Machine (SVC)": "svc"
        }
        bench_model_key = model_ui_map[bench_model_ui]
        
        plot_col1, plot_col2 = st.columns(2)
        
        with plot_col1:
            if clf.confusion_matrices and bench_model_key in clf.confusion_matrices:
                st.write("**Confusion Matrix Heatmap**")
                fig, ax = plt.subplots(figsize=(5.5, 4.5))
                cm_data = np.array(clf.confusion_matrices[bench_model_key])
                classes = clf.classes if clf.classes else ["Account Access", "Billing Issue", "General Inquiry", "Refund Request", "Technical Issue"]
                
                sns.heatmap(
                    cm_data, 
                    annot=True, 
                    fmt='d', 
                    cmap='Blues', 
                    xticklabels=classes, 
                    yticklabels=classes, 
                    ax=ax, 
                    cbar=False
                )
                plt.xticks(rotation=45, ha='right')
                plt.yticks(rotation=0)
                ax.set_ylabel("True Label", color="white")
                ax.set_xlabel("Predicted Label", color="white")
                fig.patch.set_facecolor('#0e1117')
                ax.set_facecolor('#0e1117')
                ax.tick_params(colors='white')
                ax.title.set_color('white')
                plt.tight_layout()
                st.pyplot(fig)
            else:
                st.info("No confusion matrix loaded. Please retrain models to compute.")
                
        with plot_col2:
            if clf.feature_importances and bench_model_key in clf.feature_importances:
                st.write("**Top Vocabulary Feature Importances**")
                classes = clf.classes if clf.classes else ["Account Access", "Billing Issue", "General Inquiry", "Refund Request", "Technical Issue"]
                selected_class = st.selectbox("Inspect Word Predictors for Category:", classes)
                
                features = clf.feature_importances[bench_model_key].get(selected_class, [])
                if features:
                    words = [f[0] for f in features][:12]
                    weights = [f[1] for f in features][:12]
                    
                    fig, ax = plt.subplots(figsize=(5.5, 4.5))
                    colors = sns.color_palette("plasma", len(words))
                    ax.barh(words[::-1], weights[::-1], color=colors)
                    ax.set_xlabel("TF-IDF Correlation Coefficient / Log Prob", color="white")
                    ax.set_title(f"Predictive Keywords: '{selected_class}'", color="white")
                    fig.patch.set_facecolor('#0e1117')
                    ax.set_facecolor('#0e1117')
                    ax.tick_params(colors='white')
                    ax.title.set_color('white')
                    ax.xaxis.label.set_color('white')
                    plt.tight_layout()
                    st.pyplot(fig)
                else:
                    st.info(f"No key words mapped for class '{selected_class}'")
            else:
                st.info("No feature importances loaded. Please retrain models to compute.")

    # Interactive ML Retraining Panel
    with st.container(border=True):
        st.subheader("⚙️ Live Interactive ML Training Panel")
        st.markdown("Configure classifier parameters, modify dataset split size, and retrain all models on the fly.")
    
    with st.form("interactive_train_form"):
        col_tr1, col_tr2 = st.columns(2)
        with col_tr1:
            train_size = st.slider(
                "Training Dataset Subset Size:", 
                min_value=1000, 
                max_value=50000, 
                value=25000, 
                step=1000,
                help="Higher sample sizes improve generalization, while smaller sizes train faster."
            )
            split_ratio = st.slider("Train/Test Split Ratio:", min_value=0.1, max_value=0.5, value=0.2, step=0.05)
            max_feats = st.slider("TF-IDF Max Features (Vocabulary Size):", min_value=1000, max_value=15000, value=8000, step=500)
            
        with col_tr2:
            logistic_c = st.number_input("Logistic Regression Regularization strength (C):", min_value=0.01, max_value=20.0, value=1.0, step=0.1)
            ngram_sel = st.selectbox(
                "TF-IDF Vectorization Word N-Gram Range:",
                ["(1, 1) (Unigrams only)", "(1, 2) (Unigrams + Bigrams)", "(1, 3) (Unigrams + Bigrams + Trigrams)"],
                index=1
            )
            ngram_map = {
                "(1, 1) (Unigrams only)": (1, 1),
                "(1, 2) (Unigrams + Bigrams)": (1, 2),
                "(1, 3) (Unigrams + Bigrams + Trigrams)": (1, 3)
            }
            
        submit_train = st.form_submit_button("⚡ Re-train Classifier Pipeline", width='stretch')
        
        if submit_train:
            with st.spinner("Executing pipeline: Loading dataset, cleaning text, and fitting models..."):
                try:
                    from data_loader import load_and_detect_data
                    from preprocessing import optimize_categories
                    
                    df = load_and_detect_data()
                    df = optimize_categories(df)
                    
                    if len(df) > train_size:
                        df = df.sample(train_size, random_state=42)
                        
                    metrics = clf.train(
                        df=df, 
                        test_size=split_ratio, 
                        max_features=max_feats, 
                        ngram_range=ngram_map[ngram_sel], 
                        logistic_C=logistic_c
                    )
                    clf.save()
                    
                    priority_clf.train()
                    priority_clf.save()
                    
                    st.cache_resource.clear()
                    st.success("🎉 Models + Priority Classifier trained and saved! Refreshed charts.")
                    st.rerun()
                except Exception as e:
                    st.error(f"Error during interactive training pipeline: {e}")


# --- 5. Admin Ticket Queue ---
elif app_mode == "📥 Admin Ticket Queue":
    st.markdown("### 📥 Live Support Ticket Queue & Agent Dispatcher")
    st.markdown("Review submitted voice transcripts, adjust priorities, resolve tickets, and trigger automated Text-to-Speech replies.")
    
    tickets = get_all_tickets()
    
    if not tickets:
        st.info("🎟️ No active tickets in the queue. Go to **AI Ticket Analyzer** or **Voice Call Agent** to submit some customer complaints!")
    else:
        tot = len(tickets)
        high = len([t for t in tickets if t['priority'] == 'High'])
        open_t = len([t for t in tickets if t['status'] in ['New', 'In Progress']])
        res_t = len([t for t in tickets if t['status'] == 'Resolved'])
        
        col_t1, col_t2, col_t3, col_t4 = st.columns(4)
        with col_t1:
            st.markdown(f"""
            <div style="background: rgba(99, 102, 241, 0.12); border: 1px solid rgba(99, 102, 241, 0.25); border-radius: 20px; padding: 20px; text-align: center; box-shadow: 0 4px 30px rgba(0, 0, 0, 0.3); backdrop-filter: blur(8px);">
                <span style="font-size: 2.8rem; font-weight: 800; color: #a5b4fc; display: block; margin-bottom: 2px;">{tot}</span>
                <span style="font-size: 0.85rem; color: #94a3b8; font-weight: 600; text-transform: uppercase; letter-spacing: 1px;">Total Tickets</span>
            </div>
            """, unsafe_allow_html=True)
        with col_t2:
            st.markdown(f"""
            <div style="background: rgba(239, 68, 68, 0.12); border: 1px solid rgba(239, 68, 68, 0.25); border-radius: 20px; padding: 20px; text-align: center; box-shadow: 0 4px 30px rgba(0, 0, 0, 0.3); backdrop-filter: blur(8px);">
                <span style="font-size: 2.8rem; font-weight: 800; color: #f87171; display: block; margin-bottom: 2px;">{high}</span>
                <span style="font-size: 0.85rem; color: #94a3b8; font-weight: 600; text-transform: uppercase; letter-spacing: 1px;">🚨 High Priority</span>
            </div>
            """, unsafe_allow_html=True)
        with col_t3:
            st.markdown(f"""
            <div style="background: rgba(245, 158, 11, 0.12); border: 1px solid rgba(245, 158, 11, 0.25); border-radius: 20px; padding: 20px; text-align: center; box-shadow: 0 4px 30px rgba(0, 0, 0, 0.3); backdrop-filter: blur(8px);">
                <span style="font-size: 2.8rem; font-weight: 800; color: #fbbf24; display: block; margin-bottom: 2px;">{open_t}</span>
                <span style="font-size: 0.85rem; color: #94a3b8; font-weight: 600; text-transform: uppercase; letter-spacing: 1px;">📁 Open Tickets</span>
            </div>
            """, unsafe_allow_html=True)
        with col_t4:
            st.markdown(f"""
            <div style="background: rgba(16, 185, 129, 0.12); border: 1px solid rgba(16, 185, 129, 0.25); border-radius: 20px; padding: 20px; text-align: center; box-shadow: 0 4px 30px rgba(0, 0, 0, 0.3); backdrop-filter: blur(8px);">
                <span style="font-size: 2.8rem; font-weight: 800; color: #34d399; display: block; margin-bottom: 2px;">{res_t}</span>
                <span style="font-size: 0.85rem; color: #94a3b8; font-weight: 600; text-transform: uppercase; letter-spacing: 1px;">✅ Resolved</span>
            </div>
            """, unsafe_allow_html=True)
            
        # Display Tickets Table
        with st.container(border=True):
            st.subheader("📋 Active Support Queue Records")
        
        df_disp = pd.DataFrame(tickets)
        df_disp = df_disp[[
            'id', 'timestamp', 'priority', 'status', 'actual_category', 
            'sentiment', 'transcript', 'model_used', 'predicted_category'
        ]]
        df_disp.rename(columns={
            'id': 'ID',
            'timestamp': 'Timestamp',
            'priority': 'Priority',
            'status': 'Status',
            'actual_category': 'Category/Department',
            'sentiment': 'Sentiment',
            'transcript': 'Transcript',
            'model_used': 'Model Used',
            'predicted_category': 'Original Prediction'
        }, inplace=True)
        
        st.dataframe(df_disp, use_container_width=True, hide_index=True)
        
        # Edit / Manage Panel
        with st.container(border=True):
            st.subheader("🛠️ Ticket Management Console")
        
        ticket_options = [f"Ticket #{t['id']} - [{t['priority']}] {t['actual_category']} ({t['status']})" for t in tickets]
        selected_ticket_str = st.selectbox("Choose a ticket to inspect or resolve:", ticket_options)
        
        if selected_ticket_str:
            selected_ticket_id = int(selected_ticket_str.split("Ticket #")[1].split(" - ")[0])
            t_detail = next(t for t in tickets if t['id'] == selected_ticket_id)
            
            det_col1, det_col2 = st.columns([1.5, 1])
            
            with det_col1:
                st.markdown(f"""
                <div style="background: rgba(15, 23, 42, 0.4); padding: 24px; border-radius: 18px; border: 1px solid rgba(255,255,255,0.06);">
                    <p style="color: #94a3b8; font-size: 0.85rem; margin-bottom: 5px;">SUBMITTED TIMESTAMP: {t_detail['timestamp']}</p>
                    <h5 style="margin-top: 0px; color: #f8f9fa; font-size: 1.15rem; font-weight: 600;">🗣️ Captured Transcript</h5>
                    <p style="font-size: 1.2rem; color: #e2e8f0; font-style: italic; line-height: 1.6; border-left: 3px solid #6366f1; padding-left: 15px; margin-top: 12px;">
                        "{t_detail['transcript']}"
                    </p>
                    <hr style="border-color: rgba(255,255,255,0.06);">
                    <div style="margin-top: 15px; display: flex; flex-wrap: wrap; gap: 10px; align-items: center;">
                        <span style="color: #94a3b8; font-size: 0.85rem;">Classification: </span>
                        <span class="badge badge-Category">{t_detail['predicted_category']}</span>
                        <span style="color: #94a3b8; font-size: 0.85rem;">Model: </span>
                        <span style="color: #a855f7; font-weight: 600; font-size: 0.9rem;">{t_detail['model_used'].upper()}</span>
                        <span style="color: #94a3b8; font-size: 0.85rem;">Sentiment: </span>
                        <span class="badge badge-{t_detail['sentiment']}">{t_detail['sentiment']}</span>
                    </div>
                </div>
                """, unsafe_allow_html=True)
                
            with det_col2:
                st.write("**Modify Ticket Attributes & Dispatch Resolution**")
                
                dept_opts = ["Account Access", "Billing Issue", "General Inquiry", "Refund Request", "Technical Issue"]
                cur_dept_idx = dept_opts.index(t_detail['actual_category']) if t_detail['actual_category'] in dept_opts else 2
                
                new_dept = st.selectbox("Override Department Route:", dept_opts, index=cur_dept_idx)
                
                prio_opts = ["High", "Medium", "Low"]
                cur_prio_idx = prio_opts.index(t_detail['priority']) if t_detail['priority'] in prio_opts else 1
                new_priority = st.selectbox("Override Priority Level:", prio_opts, index=cur_prio_idx)
                
                status_opts = ["New", "In Progress", "Resolved"]
                cur_status_idx = status_opts.index(t_detail['status']) if t_detail['status'] in status_opts else 0
                new_status = st.selectbox("Set Ticket Status:", status_opts, index=cur_status_idx)
                
                res_note = st.text_area("✍️ Resolution Notes (Optional):", value=t_detail['resolution_note'], placeholder="Type ticket agent resolution comments here...")
                
                col_btn1, col_btn2, col_btn3 = st.columns(3)
                
                with col_btn1:
                    save_btn = st.button("💾 Save Updates", width='stretch', type="primary")
                    if save_btn:
                        update_ticket(selected_ticket_id, new_dept, new_priority, new_status, res_note)
                        st.success(f"Ticket #{selected_ticket_id} updated successfully!")
                        st.rerun()
                        
                with col_btn2:
                    tts_btn = st.button("🗣️ Speak Resolution", width='stretch')
                    if tts_btn:
                        if res_note.strip():
                            tts_text = f"This is an automated notification from Support Desk regarding ticket number {selected_ticket_id}. Status has been set to {new_status}. Agent resolution note: {res_note}"
                        else:
                            tts_text = f"This is an automated notification from Support Desk regarding ticket number {selected_ticket_id}. Your ticket is routed to {new_dept} and is currently {new_status}."
                        
                        with st.spinner("Synthesizing speech reply..."):
                            audio_file = synthesize_speech(tts_text)
                            st.audio(audio_file, format="audio/mp3", autoplay=True)
                            
                with col_btn3:
                    del_btn = st.button("🗑️ Dismiss Ticket", width='stretch')
                    if del_btn:
                        delete_ticket(selected_ticket_id)
                        st.success(f"Ticket #{selected_ticket_id} deleted successfully!")
        st.markdown("</div>", unsafe_allow_html=True) # closing the inner HTML layout inside col1
        # st.container context ends here, indent level goes back to outer
        
        with st.expander("⚠️ Danger Zone: Clear Ticket Queue Records", expanded=False):
            st.warning("Warning: This action will permanently erase all tickets from the database.")
            clear_btn = st.button("Clear All Queue Records", type="secondary")
            if clear_btn:
                clear_all_tickets()
                st.success("All queue records deleted successfully!")
                st.rerun()
