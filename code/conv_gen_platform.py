import streamlit as st
import pandas as pd
import random
import os
import json
from openai import OpenAI
from datetime import datetime

# Import utility functions (assuming they're in the same directory)
try:
    from conv_generation_utils import (
        get_leave_reasons,
        replace_ABCD_with_name,
        populate_template,
        append_data_to_json,
    )
except ImportError:
    st.error("Cannot import conv_generation_utils. Make sure the file is in the same directory.")
    st.stop()

# Page configuration
st.set_page_config(
    page_title="Conversation Generation Platform",
    page_icon="💬",
    layout="wide"
)

# Title
st.title("💬 Conversation Generation Platform")
st.markdown("---")

th_name_list=["Somchai","Somchit","Prasoet","Sombun","Somsak","Narong","Prasit","Somphon","Nittaya","Sombat","Udom","Wichian","Amphon","Thawi","Charoen","Samran","Wichai","Sawat","Chan","Prani"]

# Sidebar for API key and language selection
with st.sidebar:
    st.header("⚙️ Configuration")
    
    # Language Mode Selection
    language_mode = st.radio(
        "🌐 Language Mode",
        options=["English", "Thai"],
        index=0,
        help="Select the language for conversation generation"
    )
    
    st.markdown("---")
    
    api_key = st.text_input("OpenAI API Key", type="password", help="Enter your OpenAI API key")
    
    if api_key:
        st.success("✓ API Key loaded")
    else:
        st.warning("⚠️ Please enter your API Key")
    
    st.markdown("---")
    st.markdown("### 📊 Output Settings")
    output_file = st.text_input("Output File", value="generated_conversations.json")

# Load datasets
@st.cache_data
def load_data():
    # Load names dataset
    script_dir = os.path.dirname(os.path.abspath(__file__))
    names_df = pd.read_csv(os.path.join(script_dir, "names_by_birth_year.csv"))
    names_df_sorted = names_df.sort_values(by="Count", ascending=False)
    top_20_count = int(len(names_df_sorted) * 0.2)
    names_df_top_20 = names_df_sorted.iloc[:top_20_count]

    # Load existing conversations for reference
    conv_df = pd.read_json(os.path.join(script_dir, "..", "dataset", "final_set", "Tactful_conv_set_merged(original-kj0).json"))

    return names_df_top_20, conv_df

try:
    names_df, conv_df = load_data()
    st.sidebar.success(f"✓ Loaded {len(names_df)} names and {len(conv_df)} conversations")
except Exception as e:
    st.error(f"Error loading data: {e}")
    st.stop()

# Extract unique values for dropdowns
emotions = sorted(conv_df['emotion'].unique())
relationships = sorted(conv_df['relationship'].unique())
situation_topics = sorted(conv_df['topic'].apply(lambda x: x['situation_topic']).unique())
real_reason_types = sorted(conv_df['real_reason_type'].unique())

# Main content area with tabs
tab1, tab2, tab3 = st.tabs(["🎯 Generate", "📋 Example Config", "📝 Preview"])

with tab1:
    st.header("Generate New Conversation")
    
    # Language mode indicator
    lang_emoji = "🇺🇸" if language_mode == "English" else "🇹🇭"
    st.info(f"{lang_emoji} Current Mode: **{language_mode}**")
    
    # Create two columns for better layout
    col1, col2 = st.columns(2)
    
    with col1:
        st.subheader("Basic Configuration")
        
        # IDs
        set_id = st.text_input("Set ID", value="2-10-5-2", help="Format: X-X-X-X")
        lie_id = st.text_input("Lie ID", value="5-2", help="Format: X-X")
        conv_id = st.number_input("Conversation ID", min_value=1, value=10)
        truth_id = st.number_input("Truth ID", min_value=1, value=1)
        
        # Type and emotion
        lie_type_options = ["altruistic_white_lies", "pareto_white_lies"]
        lie_type = st.selectbox("Lie Type", lie_type_options + ["Custom..."], index=0)
        if lie_type == "Custom...":
            lie_type = st.text_input("Enter custom lie type", value="")
        
        emotion_options = ["Custom..."] + emotions
        emotion = st.selectbox("Emotion", emotion_options, index=emotion_options.index("sad") if "sad" in emotion_options else 0)
        if emotion == "Custom...":
            emotion = st.text_input("Enter custom emotion", value="")
        
        # Falsification - Changed to support False, True, None
        falsification_options = {
            "False": False,
            "True": True,
            "None (null)": None
        }
        falsification_display = st.selectbox(
            "Falsification", 
            options=list(falsification_options.keys()),
            index=0,
            help="Select False, True, or None (null)"
        )
        falsification = falsification_options[falsification_display]
        
        multiple_liar = st.checkbox("Multiple Liar", value=True)
        
        real_reason_type_options = ["Custom..."] + [str(x) for x in real_reason_types]
        real_reason_type_selected = st.selectbox("Real Reason Type", real_reason_type_options, 
            index=real_reason_type_options.index("False") if "False" in real_reason_type_options else 0)
        if real_reason_type_selected == "Custom...":
            real_reason_type = st.text_input("Enter custom real reason type", value="")
        else:
            real_reason_type = real_reason_type_selected == "True"
    
    with col2:
        st.subheader("Scenario Details")
        
        # Scenario
        scenario = st.text_input("Scenario", value="discussing pet care")
        
        relationship_options = ["Custom..."] + relationships
        relationship = st.selectbox("Relationship", relationship_options,
            index=relationship_options.index("families (parents, one kid, aunt/uncle)") if "families (parents, one kid, aunt/uncle)" in relationship_options else 0)
        if relationship == "Custom...":
            relationship = st.text_input("Enter custom relationship", value="")
            relationship = replace_ABCD_with_name(relationship, "A", "B", "C", "D")
        
        # Character roles in relationship
        st.markdown("**Character Roles in Relationship:**")
        role_col1, role_col2 = st.columns(2)
        with role_col1:
            role_A = st.text_input("A (Liar) is:", value="parent", key="role_a")
            role_B = st.text_input("B (Target) is:", value="kid", key="role_b")
        with role_col2:
            role_C = st.text_input("C (Accomplice) is:", value="parent", key="role_c")
            role_D = st.text_input("D (Observer) is:", value="aunt/uncle", key="role_d")
        
        situation_topic_options = ["Custom..."] + situation_topics
        situation_topic = st.selectbox("Situation Topic", situation_topic_options,
            index=situation_topic_options.index("talking about what happened") if "talking about what happened" in situation_topic_options else 0)
        if situation_topic == "Custom...":
            situation_topic = st.text_input("Enter custom situation topic", value="")
            
        situation = st.text_input("Situation", value="kid is upset")
        lie_objective = st.text_input("Lie Objective", value="comfort feelings")
        
    st.markdown("---")
    
    # Lie details section
    st.subheader("Lie Details")
    col3, col4 = st.columns(2)
    
    with col3:
        real_reason_c = st.text_area("Real Reason (use A/B/C/D for names)", 
            value="child will be upset", height=80)
        truth_c = st.text_area("Truth (use A/B/C/D for names)", 
            value="pet won't return", height=80)
    
    with col4:
        lie_c = st.text_area("Lie (use A/B/C/D for names)", 
            value="pet is safe somewhere", height=80)
    
    st.markdown("---")
    
    # Character names section
    st.subheader("Character Names")
    
    col5, col6 = st.columns([3, 1])
    
    with col5:
        col_a, col_b, col_c, col_d = st.columns(4)
        
        with col_a:
            A_name = st.text_input("A (Liar)", value="")
        with col_b:
            B_name = st.text_input("B (Target)", value="")
        with col_c:
            C_name = st.text_input("C (Accomplice)", value="")
        with col_d:
            D_name = st.text_input("D (Observer)", value="")
    
    with col6:
        st.markdown("<br>", unsafe_allow_html=True)
        if st.button("🎲 Random Names", use_container_width=True):
            # Use Thai names if Thai mode is selected
            if language_mode == "Thai":
                random_names = random.sample(th_name_list, 4)
            else:
                random_names = names_df['Name'].sample(n=4).to_list()
            st.session_state.random_names = random_names
            st.rerun()
    
    # Apply random names if generated
    if 'random_names' in st.session_state:
        A_name = st.session_state.random_names[0] if not A_name else A_name
        B_name = st.session_state.random_names[1] if not B_name else B_name
        C_name = st.session_state.random_names[2] if not C_name else C_name
        D_name = st.session_state.random_names[3] if not D_name else D_name
    
    st.markdown("---")
    st.subheader("Leave Reasons")
    col7, col8 = st.columns([3, 1])
    with col7:
        col_lb, col_ld1, col_ld2 = st.columns(3)
        with col_lb:
            leave_reason_B = st.text_input("B's Leave Reason", value="")
        with col_ld1:
            leave_reason_D_1 = st.text_input("D's 1st Leave", value="")
        with col_ld2:
            leave_reason_D_2 = st.text_input("D's 2nd Leave", value="")
    
    with col8:
        st.markdown("<br>", unsafe_allow_html=True)
        if st.button("🎲 Random Reasons", use_container_width=True):
            # Pass th parameter based on language mode
            leave_reasons = get_leave_reasons(th=(language_mode == "Thai"))
            random_reasons = random.sample(leave_reasons, 3)
            st.session_state.random_reasons = random_reasons
            st.rerun()
    
    # Apply random reasons if generated
    if 'random_reasons' in st.session_state:
        leave_reason_B = st.session_state.random_reasons[0] if not leave_reason_B else leave_reason_B
        leave_reason_D_1 = st.session_state.random_reasons[1] if not leave_reason_D_1 else leave_reason_D_1
        leave_reason_D_2 = st.session_state.random_reasons[2] if not leave_reason_D_2 else leave_reason_D_2
    
    st.markdown("---")
    
    # Generate button
    col9, col10, col11 = st.columns([1, 2, 1])
    with col10:
        generate_button = st.button("🚀 Generate Conversation", 
            use_container_width=True, 
            type="primary",
            disabled=not api_key or not all([A_name, B_name, C_name, D_name, leave_reason_B, leave_reason_D_1, leave_reason_D_2]))

with tab2:
    st.header("📋 Example Configuration")
    
    st.markdown("""
    ### Example 1: Pet Care Scenario (Altruistic White Lie)
    
    **Basic Configuration:**
    - Set ID: `2-10-5-2`
    - Lie ID: `5-2`
    - Conversation ID: `10`
    - Truth ID: `1`
    - Lie Type: `altruistic_white_lies`
    - Emotion: `sad`
    - Multiple Liar: ✓ Yes
    - Falsification: `False` / `True` / `None (null)`
    
    **Scenario:**
    - Scenario: `discussing pet care`
    - Relationship: `families (parents, one kid, aunt/uncle)`
    - Character Roles:
      - A (Liar): `parent`
      - B (Target): `kid`
      - C (Accomplice): `parent`
      - D (Observer): `aunt/uncle`
    - Situation Topic: `talking about what happened`
    - Situation: `kid is upset`
    - Lie Objective: `comfort feelings`
    
    **Lie Details:**
    - Real Reason: `child will be upset`
    - Truth: `pet won't return`
    - Lie: `pet is safe somewhere`
    
    **Characters:** (Use random names or specify)
    - A (Liar): Parent 1
    - B (Target): Child
    - C (Accomplice): Parent 2
    - D (Observer): Aunt/Uncle
    
    ---
    
    ### Language Modes
    
    **English Mode:**
    - Uses English names from the database
    - Generates conversations in English
    - Uses English leave reasons
    
    **Thai Mode:**
    - Uses Thai names (Somchai, Somchit, etc.)
    - Generates conversations in Thai language
    - Uses Thai leave reasons
    
    ---
    
    ### Falsification Options
    
    - **False**: The lie does not involve falsification
    - **True**: The lie involves falsification
    - **None (null)**: Falsification status is not applicable or unknown
    """)
    
    st.info("💡 Tip: Use A, B, C, D as placeholders in lie details. They will be replaced with actual names.")

with tab3:
    st.header("📝 Generated Conversation Preview")
    
    # Load existing conversations from JSON file
    @st.cache_data
    def load_generated_conversations(file_path):
        try:
            if os.path.exists(file_path):
                with open(file_path, 'r', encoding='utf-8') as f:
                    data = json.load(f)
                    # Handle both list and single object formats
                    if isinstance(data, list):
                        return data
                    else:
                        return [data]
            return []
        except Exception as e:
            st.error(f"Error loading conversations: {e}")
            return []
    
    # File selector
    col_preview1, col_preview2 = st.columns([3, 1])
    
    with col_preview1:
        json_file_path = st.text_input(
            "JSON File Path", 
            value="generated_conversations.json",
            help="Path to the generated conversations JSON file"
        )
    
    with col_preview2:
        st.markdown("<br>", unsafe_allow_html=True)
        refresh_button = st.button("🔄 Refresh", use_container_width=True)
    
    if refresh_button:
        st.cache_data.clear()
    
    # Load conversations
    conversations = load_generated_conversations(json_file_path)
    
    if conversations:
        st.success(f"✓ Loaded {len(conversations)} conversation(s)")
        
        # Create conversation selector
        conv_options = []
        for i, conv in enumerate(conversations):
            conv_id = conv.get('conv_id', i)
            set_id = conv.get('set_id', 'N/A')
            lie_type = conv.get('lie_type', 'N/A')
            emotion = conv.get('emotion', 'N/A')
            lang = conv.get('language', 'N/A')
            
            label = f"Conv {conv_id} | Set {set_id} | {lie_type} | {emotion} | {lang}"
            conv_options.append((label, i))
        
        selected_label = st.selectbox(
            "Select Conversation",
            options=[opt[0] for opt in conv_options],
            index=len(conv_options) - 1  # Default to most recent
        )
        
        # Get selected conversation
        selected_index = next(i for label, i in conv_options if label == selected_label)
        conv = conversations[selected_index]
        
        st.markdown("---")
        
        # Show timestamp and language if available
        info_parts = []
        if 'timestamp' in conv:
            info_parts.append(f"🕒 Generated at: {conv['timestamp']}")
        if 'language' in conv:
            lang_emoji = "🇺🇸" if conv['language'] == "English" else "🇹🇭"
            info_parts.append(f"{lang_emoji} Language: {conv['language']}")
        
        if info_parts:
            st.info(" | ".join(info_parts))
        
        # Show metadata
        with st.expander("📊 Metadata", expanded=False):
            metadata = {
                "set_id": conv.get('set_id', 'N/A'),
                "lie_id": conv.get('lie_id', 'N/A'),
                "conv_id": conv.get('conv_id', 'N/A'),
                "truth_id": conv.get('truth_id', 'N/A'),
                "lie_type": conv.get('lie_type', 'N/A'),
                "emotion": conv.get('emotion', 'N/A'),
                "language": conv.get('language', 'N/A'),
                "relationship": conv.get('relationship', 'N/A'),
                "multiple_liar": conv.get('multiple_liar', 'N/A'),
                "real_reason_type": conv.get('real_reason_type', 'N/A'),
                "characters": conv.get('characters', {}),
                "character_roles": conv.get('character_roles', {}),
                "topic": conv.get('topic', {}),
                "lie": conv.get('lie', {}),
            }
            
            if 'full_context_tokens' in conv:
                metadata["tokens"] = {
                    "full_context": conv.get('full_context_tokens', 0),
                    "short_context": conv.get('short_context_tokens', 0)
                }
            
            st.json(metadata)
        
        # Show conversation parts
        if 'part_1' in conv:
            st.subheader("Part 1: Initial Conversation")
            st.text_area("", value=conv['part_1'], height=200, disabled=True, key=f"preview_part1_{selected_index}")
        
        if 'part_2' in conv:
            st.subheader("Part 2: B & D Leave")
            st.text_area("", value=conv['part_2'], height=200, disabled=True, key=f"preview_part2_{selected_index}")
        
        if 'part_3' in conv:
            st.subheader("Part 3: D Returns")
            st.text_area("", value=conv['part_3'], height=200, disabled=True, key=f"preview_part3_{selected_index}")
        
        if 'part_4' in conv:
            st.subheader("Part 4: B Returns & Lie")
            st.text_area("", value=conv['part_4'], height=200, disabled=True, key=f"preview_part4_{selected_index}")
        
        # Show full context if available
        if 'full_context' in conv:
            with st.expander("📄 Full Context", expanded=False):
                st.text_area("", value=conv['full_context'], height=400, disabled=True, key=f"preview_full_{selected_index}")
        
        # Download button
        col_dl1, col_dl2, col_dl3 = st.columns([1, 2, 1])
        with col_dl2:
            st.download_button(
                label="📥 Download This Conversation",
                data=json.dumps(conv, indent=2, ensure_ascii=False),
                file_name=f"conversation_{conv.get('conv_id', selected_index)}.json",
                mime="application/json",
                use_container_width=True
            )
    
    elif 'generated_conversation' in st.session_state:
        # Fallback to session state if file doesn't exist
        conv = st.session_state.generated_conversation
        
        st.success(f"✓ Generated at {conv.get('timestamp', 'Unknown')}")
        st.warning("⚠️ Showing conversation from current session only. Check file path to load saved conversations.")
        
        # Show metadata
        with st.expander("📊 Metadata", expanded=False):
            st.json({
                "set_id": conv['set_id'],
                "lie_id": conv['lie_id'],
                "conv_id": conv['conv_id'],
                "truth_id": conv['truth_id'],
                "lie_type": conv['lie_type'],
                "emotion": conv['emotion'],
                "language": conv.get('language', 'N/A'),
                "characters": conv['characters'],
                "tokens": {
                    "full_context": conv['full_context_tokens'],
                    "short_context": conv['short_context_tokens']
                }
            })
        
        # Show conversation parts
        st.subheader("Part 1: Initial Conversation")
        st.text_area("", value=conv['part_1'], height=200, disabled=True, key="preview_part1")
        
        st.subheader("Part 2: B & D Leave")
        st.text_area("", value=conv['part_2'], height=200, disabled=True, key="preview_part2")
        
        st.subheader("Part 3: D Returns")
        st.text_area("", value=conv['part_3'], height=200, disabled=True, key="preview_part3")
        
        st.subheader("Part 4: B Returns & Lie")
        st.text_area("", value=conv['part_4'], height=200, disabled=True, key="preview_part4")
        
        # Download button
        st.download_button(
            label="📥 Download JSON",
            data=json.dumps(conv, indent=2, ensure_ascii=False),
            file_name=f"conversation_{conv['conv_id']}.json",
            mime="application/json"
        )
    else:
        st.info("📁 No conversations found. Either generate a new conversation or check the JSON file path above.")
        st.markdown("""
        **Tips:**
        - Generate a conversation in the **Generate** tab
        - Or provide the path to an existing JSON file with generated conversations
        - Click the 🔄 Refresh button after generating new conversations
        """)

# Generation logic
if generate_button:
    with st.spinner("🤖 Generating conversation..."):
        try:
            # Initialize OpenAI client
            client = OpenAI(api_key=api_key)
            
            # Replace placeholders in lie details
            real_reason_c_filled = replace_ABCD_with_name(real_reason_c, A_name, B_name, C_name, D_name)
            lie_c_filled = replace_ABCD_with_name(lie_c, A_name, B_name, C_name, D_name)
            truth_c_filled = replace_ABCD_with_name(truth_c, A_name, B_name, C_name, D_name)
            situation_topic=replace_ABCD_with_name(situation_topic, A_name, B_name, C_name, D_name)
            situation=replace_ABCD_with_name(situation, A_name, B_name, C_name, D_name)
            scenario=replace_ABCD_with_name(scenario, A_name, B_name, C_name, D_name)
            lie_objective=replace_ABCD_with_name(lie_objective, A_name, B_name, C_name, D_name)
            relationship=replace_ABCD_with_name(relationship, A_name, B_name, C_name, D_name)
            
            # Build relationship descriptor
            relationship_descriptor = f"{A_name} is {role_A}, {B_name} is {role_B}, {C_name} is {role_C}, and {D_name} is {role_D}"
            
            # Define templates
            template_step_1 = f"""In this conversation, four characters are participating: {A_name} , {B_name} , {C_name} , and {D_name}.

Start the conversation with all participants {scenario}.

They are {relationship}, where {relationship_descriptor}.

Generate a natural conversation between all four characters. Use their names and format the dialogue clearly."""

            template_step_2 = f"""Now, {B_name} leaves the conversation because of the reason: {leave_reason_B}, while {D_name} leaves the conversation because of the reason: {leave_reason_D_1}.

With {B_name} and {D_name} absent, {A_name} and {C_name} continue their conversation, shifting the topic to '{real_reason_c_filled}'. Ensure the dialogue flows naturally without directly repeating these exact words.

Remember to indicate that {B_name} and {D_name} are leaving the conversation."""

            template_step_3 = f"""Now, only {D_name} returns and rejoins the conversation after previously leaving. Note that {B_name} has not returned to the conversation yet.

Then the conversation continues and naturally shifts to {situation_topic}. Do NOT catch up or recap details from the previous conversation.

And then '{truth_c_filled}' is naturally revealed in the conversation.

After engaging briefly, {D_name} leaves the conversation again because of the reason: {leave_reason_D_2}.

IMPORTANT:
• Clearly indicate through dialogue only (e.g., "I'm back," or "I have to leave again now.") that {D_name} first returns and later departs again.
• Do NOT use third-person narrative or stage directions.
• Avoid mentioning details from previous conversations."""

            template_step_4 = f"""Now {B_name} returns to the conversation after leaving the conversation.

First have {B_name} explicitly indicate the return through dialogue. Do NOT catch up or recap details from the previous conversation.

And then {situation} naturally unfolds. Make sure the dialogue flows naturally without directly repeating these exact words.

In response, {A_name} and {C_name} tells {B_name} that '{lie_c_filled}' to {lie_objective}.

IMPORTANT:
• Do NOT use third-person narrative or stage directions.
• Avoid mentioning details from previous conversations."""

            # Fill templates
            filled_step_1 = template_step_1
            filled_step_2 = template_step_2
            filled_step_3 = template_step_3
            filled_step_4 = template_step_4
            
            # System prompt based on language mode
            if language_mode == "Thai":
                system_prompt = "You are a creative writer generating natural, realistic conversations in Thai language. Write dialogue in a clear format with character names followed by their speech. keep the name english. Use natural Thai conversation style."
            else:
                system_prompt = "You are a creative writer generating natural, realistic conversations. Write dialogue in a clear format with character names followed by their speech."
            
            # Helper function for generation
            def generate_conversation_part(prompt, previous_context=""):
                messages = [
                    {
                        "role": "system",
                        "content": system_prompt
                    }
                ]
                
                if previous_context:
                    messages.append({
                        "role": "user",
                        "content": f"Previous conversation:\n{previous_context}\n\n{prompt}"
                    })
                else:
                    messages.append({
                        "role": "user",
                        "content": prompt
                    })
                
                response = client.chat.completions.create(
                    model="gpt-4o-2024-08-06",
                    messages=messages,
                    temperature=0.2,
                    max_tokens=1000
                )
                
                tokens = response.usage.completion_tokens
                return response.choices[0].message.content, tokens
            
            # Generate parts with progress
            progress_bar = st.progress(0)
            status_text = st.empty()
            
            status_text.text("Generating Part 1...")
            part_1, tokens1 = generate_conversation_part(filled_step_1)
            progress_bar.progress(25)
            
            status_text.text("Generating Part 2...")
            part_2, tokens2 = generate_conversation_part(filled_step_2, part_1)
            progress_bar.progress(50)
            
            status_text.text("Generating Part 3...")
            part_3, tokens3 = generate_conversation_part(filled_step_3, f"{part_1}\n\n{part_2}")
            progress_bar.progress(75)
            
            status_text.text("Generating Part 4...")
            part_4, tokens4 = generate_conversation_part(filled_step_4, f"{part_1}\n\n{part_2}\n\n{part_3}")
            progress_bar.progress(100)
            
            # Combine results
            full_context = "\n\n".join([part_1, part_2, part_3, part_4])
            full_context_tokens = tokens1 + tokens2 + tokens3 + tokens4
            short_context = "\n\n".join([part_2, part_3, part_4])
            short_context_tokens = tokens2 + tokens3 + tokens4
            
            data_dict = {
                "set_id": set_id,
                "lie_id": lie_id,
                "conv_id": conv_id,
                "truth_id": truth_id,
                "lie_type": lie_type,
                "emotion": emotion,
                "topic": {
                    "scenario": scenario,
                    "situation_topic": situation_topic,
                    "situation": situation,
                    "lie_objective": lie_objective,
                    "leave_reason_B": leave_reason_B,
                    "leave_reason_D_1": leave_reason_D_1,
                    "leave_reason_D_2": leave_reason_D_2
                },
                "relationship": relationship,
                "multiple_liar": multiple_liar,
                "real_reason_type": real_reason_type,
                "characters": {
                    "liar": A_name,
                    "target": B_name,
                    "accomplice": C_name,
                    "observer": D_name
                },
                "character_roles": {
                    "liar": role_A,
                    "target": role_B,
                    "accomplice": role_C,
                    "observer": role_D
                },
                "lie": {
                    "real_reason_c": real_reason_c_filled,
                    "lie_c": lie_c_filled,
                    "truth_c": truth_c_filled,
                    "falsification": falsification
                },
                "part_1": part_1,
                "part_2": part_2,
                "part_3": part_3,
                "part_4": part_4,
                "full_context": full_context,
                "short_context": short_context,
                "full_context_tokens": full_context_tokens,
                "short_context_tokens": short_context_tokens,
                "timestamp": datetime.now().isoformat()
            }
            
            # Save to file
            append_data_to_json(data_dict, output_file)
            
            # Store in session state
            st.session_state.generated_conversation = data_dict
            
            status_text.empty()
            progress_bar.empty()
            
            st.success(f"✅ Conversation generated successfully! ({full_context_tokens} tokens)")
            st.info(f"💾 Saved to {output_file}")
            st.balloons()
            
            # Switch to preview tab
            st.rerun()
            
        except Exception as e:
            st.error(f"❌ Error generating conversation: {str(e)}")
            st.exception(e)