import streamlit as st
import os
import tempfile
import time
from speech_to_text import transcribe_audio
from text_to_speech import synthesize_speech
from preprocessing import get_sentiment_and_priority
from model import TicketClassifier

# Page Config must be the first Streamlit command
st.set_page_config(page_title="Voice Support AI", page_icon="🎧", layout="wide", initial_sidebar_state="expanded")

# Inject Modern CSS for a Premium Aesthetic
st.markdown("""
<style>
    @import url('https://fonts.googleapis.com/css2?family=Outfit:wght@300;500;700&display=swap');
    
    html, body, [class*="css"]  {
        font-family: 'Outfit', sans-serif !important;
    }
    
    .main-title {
        background: -webkit-linear-gradient(45deg, #6C63FF, #FF6584);
        -webkit-background-clip: text;
        -webkit-text-fill-color: transparent;
        font-weight: 700;
        font-size: 3.5rem;
        padding-bottom: 0px;
        margin-bottom: -15px;
        text-align: center;
    }
    
    .sub-title {
        text-align: center;
        color: #A0AEC0;
        font-size: 1.25rem;
        margin-bottom: 3rem;
        font-weight: 300;
    }
    
    .glass-card {
        background: rgba(255, 255, 255, 0.03);
        border: 1px solid rgba(255, 255, 255, 0.1);
        border-radius: 16px;
        padding: 24px;
        margin-top: 10px;
        box-shadow: 0 8px 32px 0 rgba(0, 0, 0, 0.3);
        backdrop-filter: blur(10px);
        -webkit-backdrop-filter: blur(10px);
    }
    
    .badge {
        padding: 6px 14px;
        border-radius: 8px;
        font-weight: 700;
        font-size: 0.95rem;
        display: inline-block;
        letter-spacing: 0.5px;
    }
    
    .badge-High { background-color: rgba(255, 75, 75, 0.15); color: #ff6b6b; border: 1px solid rgba(255, 75, 75, 0.4); }
    .badge-Medium { background-color: rgba(255, 171, 0, 0.15); color: #ffbf00; border: 1px solid rgba(255, 171, 0, 0.4); }
    .badge-Low { background-color: rgba(0, 200, 83, 0.15); color: #00e676; border: 1px solid rgba(0, 200, 83, 0.4); }
    
    .badge-Negative { background-color: rgba(255, 75, 75, 0.15); color: #ff6b6b; border: 1px solid rgba(255, 75, 75, 0.4); }
    .badge-Neutral { background-color: rgba(160, 174, 192, 0.15); color: #cbd5e1; border: 1px solid rgba(160, 174, 192, 0.4); }
    .badge-Positive { background-color: rgba(0, 200, 83, 0.15); color: #00e676; border: 1px solid rgba(0, 200, 83, 0.4); }
    
    .badge-Category { background-color: rgba(108, 99, 255, 0.15); color: #a29bfe; border: 1px solid rgba(108, 99, 255, 0.4); }
</style>
""", unsafe_allow_html=True)

st.markdown('<div class="main-title">Voice Support AI</div>', unsafe_allow_html=True)
st.markdown('<div class="sub-title">Instantly convert voice queries into categorized tickets, extracting sentiment & predicting case urgency.</div>', unsafe_allow_html=True)
st.markdown("---")

st.sidebar.title("🧭 Navigation")
app_mode = st.sidebar.radio("Go to:", ["🏢 Departments Overview", "🎙️ AI Ticket Analyzer", "📞 Voice Call Agent"])
st.sidebar.markdown("---")

@st.cache_resource(show_spinner="Loading ML Artifacts...")
def load_classifier():
    clf = TicketClassifier()
    loaded = clf.load()
    return clf, loaded

clf, is_loaded = load_classifier()

def render_analysis_card(transcribed_text):
    category = clf.predict(transcribed_text)
    sentiment, priority = get_sentiment_and_priority(transcribed_text)
    
    dept_mapping = {
        "Billing Issue": "Finance & Billing",
        "Refund Request": "Finance & Billing",
        "Technical Issue": "IT & Engineering",
        "Account Access": "Security & Auth",
        "General Inquiry": "Tier 1 Support"
    }
    routed_dept = dept_mapping.get(category, "Tier 1 Support")
    
    res_col1, res_col2 = st.columns([1.5, 1])
    with res_col1:
        st.markdown(f"""
        <div class="glass-card">
            <h4 style="margin-top:0px; color: #f8f9fa;">🗣️ Official Call Transcript</h4>
            <p style="font-size: 1.1rem; color: #cbd5e1; line-height: 1.6; font-style:italic;">"{transcribed_text}"</p>
        </div>
        """, unsafe_allow_html=True)
        
    with res_col2:
        st.markdown(f"""
        <div class="glass-card">
            <h4 style="margin-top:0px; color: #f8f9fa;">🎛️ AI Routing Analysis</h4>
            <br>
            <div>
                <span style="color: #94a3b8; font-size: 0.9rem;">Routed Department</span><br>
                <span class="badge" style="background-color:rgba(255,101,132,0.2); color:#FF6584; border:1px solid #FF6584;">{routed_dept}</span>
            </div>
            <br>
            <div>
                <span style="color: #94a3b8; font-size: 0.9rem;">AI Classification</span><br>
                <span class="badge badge-Category">{category}</span>
            </div>
            <br>
            <div>
                <span style="color: #94a3b8; font-size: 0.9rem;">Customer Tone</span><br>
                <span class="badge badge-{sentiment}">{sentiment}</span>
            </div>
            <br>
            <div>
                <span style="color: #94a3b8; font-size: 0.9rem;">Urgency Level</span><br>
                <span class="badge badge-{priority}">{priority} priority</span>
            </div>
        </div>
        """, unsafe_allow_html=True)


if app_mode == "🏢 Departments Overview":
    st.markdown("### 🏦 Intelligent Ticket Routing Architecture", unsafe_allow_html=True)
    st.info("👈 **Select a new AI tool from the sidebar to continue!**")
    
    col1, col2 = st.columns(2)
    with col1:
        st.markdown("""
        <div class="glass-card" style="text-align:center;">
            <div style="font-size:3rem;">💳</div>
            <h4 style="color:#FF6584;">Finance & Billing</h4>
            <p style="color:#a29bfe;">Handles: Billing / Refunds</p>
        </div>
        <div class="glass-card" style="text-align:center;">
            <div style="font-size:3rem;">🧑‍💻</div>
            <h4 style="color:#FF6584;">IT & Engineering</h4>
            <p style="color:#a29bfe;">Handles: Technical Issues</p>
        </div>
        """, unsafe_allow_html=True)
    with col2:
        st.markdown("""
        <div class="glass-card" style="text-align:center;">
            <div style="font-size:3rem;">🔐</div>
            <h4 style="color:#FF6584;">Security & Auth</h4>
            <p style="color:#a29bfe;">Handles: Account Access</p>
        </div>
        <div class="glass-card" style="text-align:center; border-color:#6C63FF;">
            <div style="font-size:3rem;">⚡</div>
            <h4 style="color:#6C63FF;">Priority Escalation</h4>
            <p style="color:#a29bfe;">Trigger: High Priority Analysis</p>
        </div>
        """, unsafe_allow_html=True)

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
            text_input = st.text_area("✍️ Customer Message Context", height=150)

        analyze_btn = st.button("🚀 Analyze Support Ticket", use_container_width=True, type="primary")

    if analyze_btn:
        if not audio_path and not text_input.strip():
            st.warning("⚠️ Please provide an audio recording, file, or text input first.")
        else:
            with st.spinner("🤖 Whisper is transcribing & ML is analyzing..."):
                if audio_path:
                    transcribed_text = transcribe_audio(audio_path)
                    try:
                        os.remove(audio_path)
                    except Exception:
                        pass
                else:
                    transcribed_text = text_input

                if transcribed_text.strip():
                    render_analysis_card(transcribed_text)
                else:
                    st.error("Input was empty or unintelligible.")

elif app_mode == "📞 Voice Call Agent":
    if not is_loaded:
        st.error("🚨 ML Models Missing! Train model first.")
        st.stop()
        
    st.markdown("### 📞 Real-Time Live Phone Call Support Simulation")
    st.markdown("Experience a two-way AI voice call! Just **speak to the Agent**, and the Agent will **talk back** to you. Once the interaction is resolved, the entire audio record is evaluated by our ML logic.")
    
    if "messages" not in st.session_state:
        st.session_state.messages = []
        st.session_state.chat_finished = False
        greeting = "Hello! Welcome to our Support Desk. I am your AI agent. Please speak clearly using your microphone, and tell me: How can I assist you today?"
        st.session_state.messages.append({"role": "assistant", "content": greeting})
        st.session_state.pending_audio = synthesize_speech(greeting)

    # Initialize audio cache tracker
    if "last_processed_audio_hash" not in st.session_state:
        st.session_state.last_processed_audio_hash = None

    # Render Visual Conversation History
    for msg in st.session_state.messages:
        with st.chat_message(msg["role"]):
            st.write(msg["content"])
            
    # Autoplay pending AI voice if exists
    if "pending_audio" in st.session_state and st.session_state.pending_audio is not None:
        st.audio(st.session_state.pending_audio, format="audio/mp3", autoplay=True)
        # Clear it so it doesn't replay when the user talks back
        st.session_state.pending_audio = None
            
    if not st.session_state.chat_finished:
        st.markdown("---")
        # Voice Input Component
        audio_value = st.audio_input("🗣️ It's your turn to speak...")
        
        if audio_value:
            audio_bytes = audio_value.getbuffer().tobytes()
            audio_hash = hash(audio_bytes)
            
            # Make sure we only process this recording ONCE to prevent infinite looping
            if st.session_state.last_processed_audio_hash != audio_hash:
                st.session_state.last_processed_audio_hash = audio_hash
                
                with st.spinner("Agent is listening & transcribing..."):
                    with tempfile.NamedTemporaryFile(delete=False, suffix=".wav") as f:
                        f.write(audio_bytes)
                        audio_path = f.name
                        
                    user_text = transcribe_audio(audio_path)
                    os.remove(audio_path)
                    
                    st.session_state.messages.append({"role": "user", "content": user_text})
                    st.rerun()  # Instantly display user message
            else:
                pass # Already processed this specific buffer
                
        # If the last message was from the User, explicitly Generate AI Response!
        if st.session_state.messages[-1]["role"] == "user":
            with st.spinner("Agent is thinking and generating a voice reply..."):
                time.sleep(1) # Fake realistic latency
                user_msgs = len([m for m in st.session_state.messages if m["role"] == "user"])
                
                if user_msgs == 1:
                    reply = "I'm sorry to hear you're dealing with that. Could you provide a little more context using your microphone so I can be absolutely sure about how to route your ticket?"
                else:
                    reply = "Thank you so much for securely providing those details! Our call is now concluded. I will immediately escalate this transcript to the optimal human department right away. Have a nice day!"
                    st.session_state.chat_finished = True
                    
                st.session_state.messages.append({"role": "assistant", "content": reply})
                st.session_state.pending_audio = synthesize_speech(reply) # Generate Audio!
                st.rerun()

    else:
        st.success("☎️ **Call Dropped.** Conversation Archived. Generating Live ML Routing Trace...")
        
        # Compile user context to analyze
        full_transcript = " ".join([m["content"] for m in st.session_state.messages if m["role"] == "user"])
        
        render_analysis_card(full_transcript)
        
        if st.button("🔄 Initiate New Call"):
            del st.session_state.messages
            del st.session_state.chat_finished
            st.session_state.last_processed_audio_hash = None
            st.rerun()
