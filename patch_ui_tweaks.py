with open("app.py", "r", encoding="utf-8") as f:
    content = f.read()

# --- Change 1: Update heading text ---
old_heading = '''    <h1 style="font-family: 'Playfair Display', serif; background: linear-gradient(90deg, #D4AF37, #F4E4BC); -webkit-background-clip: text; -webkit-text-fill-color: transparent; font-size: 44px; margin-bottom: 0;">
        ⚖️ Nyaya AI
    </h1>'''
new_heading = '''    <h1 style="font-family: 'Playfair Display', serif; background: linear-gradient(90deg, #D4AF37, #F4E4BC); -webkit-background-clip: text; -webkit-text-fill-color: transparent; font-size: 44px; margin-bottom: 0;">
        ⚖️ Nyaya AI / Legal AI
    </h1>'''

if old_heading in content:
    content = content.replace(old_heading, new_heading, 1)
    print("Heading updated.")
else:
    print("WARNING: heading anchor not found.")

# --- Change 2: Remove the centered mic-toggle block above chat_input ---
old_mic_block = '''# --- Mic toggle sits directly above the input bar, ChatGPT-style ---
mic_col1, mic_col2, mic_col3 = st.columns([5, 1, 5])
with mic_col2:
    if st.button("🎤", key="mic_toggle", help="Ask by voice"):
        st.session_state.show_mic = not st.session_state.show_mic

if st.session_state.show_mic:
    audio_input = st.audio_input("Record your question", label_visibility="collapsed")
    if audio_input is not None:
        audio_id = audio_input.file_id if hasattr(audio_input, "file_id") else str(len(audio_input.getvalue()))
        if audio_id != st.session_state.last_audio_id:
            st.session_state.last_audio_id = audio_id
            with st.spinner("Transcribing your voice..."):
                transcribed = transcribe_audio(audio_input)
            if transcribed:
                st.success(f"Heard: \\"{transcribed}\\"")
                st.session_state.pending_query = transcribed
                st.session_state.show_mic = False
                st.rerun()
            else:
                st.warning("Could not understand the audio. Please try again or type your question.")

'''

if old_mic_block in content:
    content = content.replace(old_mic_block, "", 1)
    print("Removed top-level mic block.")
else:
    print("WARNING: top-level mic block not found.")

# --- Change 3: Add mic back as an expander inside tab_chat ---
old_tab_start = '''with tab_chat:
    if not st.session_state.messages:'''
new_tab_start = '''with tab_chat:
    with st.expander("🎤 Ask by voice"):
        audio_input = st.audio_input("Record your question")
        if audio_input is not None:
            audio_id = audio_input.file_id if hasattr(audio_input, "file_id") else str(len(audio_input.getvalue()))
            if audio_id != st.session_state.last_audio_id:
                st.session_state.last_audio_id = audio_id
                with st.spinner("Transcribing your voice..."):
                    transcribed = transcribe_audio(audio_input)
                if transcribed:
                    st.success(f"Heard: \\"{transcribed}\\"")
                    st.session_state.pending_query = transcribed
                    st.rerun()
                else:
                    st.warning("Could not understand the audio. Please try again or type your question.")

    if not st.session_state.messages:'''

if old_tab_start in content:
    content = content.replace(old_tab_start, new_tab_start, 1)
    print("Added expander mic inside Chat tab.")
else:
    print("WARNING: tab_chat start anchor not found.")

with open("app.py", "w", encoding="utf-8") as f:
    f.write(content)
