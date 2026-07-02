import os
import time
import streamlit as st
import pandas as pd
from langchain_core.messages import HumanMessage, AIMessage

from agent_backend import (
    director_agent, 
    director_agent_fallback, 
    config, 
    validate_user_input, 
    log_tokens, 
    CSV_LOG_FILE
)

st.set_page_config(layout="wide", page_title="AI Analytical Workbench")
st.title("📊 Enterprise Multi-Agent Analytics Studio")
st.caption("Director-Worker Hierarchical Architecture with Autoregressive Token Streaming")
st.markdown("---")

if "messages" not in st.session_state:
    st.session_state.messages = []
if "failover_triggered" not in st.session_state:
    st.session_state.failover_triggered = False

SUPPORTED_EXTENSIONS = ('.csv', '.json', '.parquet')

local_files = [
    f for f in os.listdir('.') 
    if f.endswith(SUPPORTED_EXTENSIONS) and f != "agent_tokens_logs.csv" and not f.startswith("~")
]

def token_streamer(text_content: str, delay: float = 0.02):
    """Yields chunks of text sequentially to simulate an autoregressive streaming effect."""
    for word in text_content.split(" "):
        yield word + " "
        time.sleep(delay)

# ==========================================
# STATE-SYNCHRONIZED SIDEBAR CONTROL PANEL
# ==========================================
with st.sidebar:
    st.header("📂 Data Ingestion Controller")
    source_mode = st.radio("Select Data Input Source Mode:", ["Choose Local Workspace File", "Upload New File Asset"])
    active_file_path = None
    
    if source_mode == "Choose Local Workspace File":
        if local_files:
            if "workspace_dataframe_viewer" not in st.session_state:
                st.session_state.workspace_dataframe_viewer = local_files[0]
                
            selected_filename = st.selectbox(
                "Select target active dataset file:", 
                local_files, 
                key="sidebar_file_selector"
            )
            
            st.session_state.workspace_dataframe_viewer = selected_filename
            active_file_path = selected_filename
            st.success(f"Selected Target: `{active_file_path}`")
        else:
            st.warning("No standard data frames detected inside local workspace directories.")
            
    else:
        uploaded_file = st.file_uploader("Drag and drop your file here:", type=["csv", "json", "parquet"])
        if uploaded_file is not None:
            active_file_path = uploaded_file.name
            with open(active_file_path, "wb") as f:
                f.write(uploaded_file.getbuffer())
            st.success(f"Uploaded and staged: `{active_file_path}`")
            st.session_state.workspace_dataframe_viewer = active_file_path

# ==========================================
# MAIN INTERFACE TWO COLUMN LAYOUT MATRIX
# ==========================================
terminal_col, graphic_col = st.columns([1, 1])

with terminal_col:
    st.subheader("Supervisor Console Interface")
    
    for interaction in st.session_state.messages:
        with st.chat_message(interaction["role"]):
            st.markdown(interaction["content"])

    if user_prompt := st.chat_input("Enter formatting, mathematical or visualization requests..."):
        if not active_file_path:
            st.warning("⚠️ Access Denied: Please select a file target in the sidebar first.")
        else:
            contextualized_prompt = f"Using the dataset file '{active_file_path}', please execute the following instruction: {user_prompt}"
            
            st.chat_message("user").write(user_prompt)
            st.session_state.messages.append({"role": "user", "content": user_prompt})
            
            is_safe, error_message = validate_user_input(user_prompt, max_token_count=100)
            
            if not is_safe:
                st.error(error_message)
                st.session_state.messages.append({"role": "assistant", "content": f"Blocked: {error_message}"})
            else:
                with st.chat_message("assistant"):
                    trace_container = st.container()
                    with trace_container:
                        status_loader = st.empty()
                        status_loader.markdown("⏳ **System Initializing...** Assembling multi-agent execution graphs...")
                    
                    runtime_config = {
                        "configurable": config["configurable"],
                        "recursion_limit": 100  
                    }
                    
                    payload = {"messages": [HumanMessage(content=contextualized_prompt)]}
                    last_seen_message = None
                    
                    try:
                        selected_agent = director_agent_fallback if st.session_state.failover_triggered else director_agent
                        
                        for chunk in selected_agent.stream(payload, config=runtime_config, stream_mode="updates"):
                            for node_name, node_update in chunk.items():
                                if "messages" in node_update:
                                    last_msg = node_update["messages"][-1]
                                    
                                    if hasattr(last_msg, "tool_calls") and last_msg.tool_calls:
                                        for tcall in last_msg.tool_calls:
                                            status_loader.markdown(f"⚙️ **Director Action**: Activating module `{tcall['name']}`...")
                                    elif node_name == "tools":
                                        status_loader.markdown("🔄 **Tool Sync**: Committing changes back onto disk targets...")
                                    else:
                                        status_loader.markdown(f"🧠 **Thinking States**: Processing tokens inside node: `{node_name}`...")
                        
                        status_loader.empty()
                        
                        final_state = selected_agent.get_state(runtime_config)
                        if final_state.values and "messages" in final_state.values:
                            last_seen_message = final_state.values["messages"][-1]
                        
                        if last_seen_message and last_seen_message.content:
                            final_text = str(last_seen_message.content).strip()
                            if not final_text or final_text.startswith("{"):
                                final_text = "Process completed successfully. Review modifications inside the Live Workspace File Inspector below."
                                
                            streamed_response = st.write_stream(token_streamer(final_text, delay=0.02))
                            st.session_state.messages.append({"role": "assistant", "content": streamed_response})
                            if isinstance(last_seen_message, AIMessage):
                                log_tokens(last_seen_message)
                                
                    except Exception as api_outage:
                        status_loader.warning(f"🔧 Primary exception: {str(api_outage)}. Running failover cluster...")
                        st.session_state.failover_triggered = True
                        
                        try:
                            for chunk in director_agent_fallback.stream(payload, config=runtime_config, stream_mode="updates"):
                                for node_name, node_update in chunk.items():
                                    if "messages" in node_update:
                                        last_msg = node_update["messages"][-1]
                            
                            status_loader.empty()
                            final_state = director_agent_fallback.get_state(runtime_config)
                            if final_state.values and "messages" in final_state.values:
                                last_seen_message = final_state.values["messages"][-1]
                                
                            if last_seen_message and last_seen_message.content:
                                final_text = str(last_seen_message.content).strip()
                                streamed_response = st.write_stream(token_streamer(final_text, delay=0.02))
                                st.session_state.messages.append({"role": "assistant", "content": streamed_response})
                                if isinstance(last_seen_message, AIMessage):
                                    log_tokens(last_seen_message)
                        except Exception as fatal_err:
                            status_loader.empty()
                            st.error(f"Execution Error: {str(fatal_err)}")
                            st.session_state.messages.append({"role": "assistant", "content": str(fatal_err)})
                            
                st.rerun()

with graphic_col:
    st.subheader("Data Visualization Display Output")
    if os.path.exists("data_analyzer.png"):
        st.image("data_analyzer.png", use_container_width=True, caption="Pipeline Chart Output Canvas")
        if st.button("Flush Visual Workspace Output Matrix Cache"):
            os.remove("data_analyzer.png")
            st.rerun()
    else:
        st.info("Awaiting structural chart generation operations on current transaction execution streams.")
        
    st.markdown("---")
    st.subheader("Telemetry Log Audit Ledger (CSV Overview)")
    if os.path.exists(CSV_LOG_FILE):
        log_df = pd.read_csv(CSV_LOG_FILE)
        st.dataframe(log_df.tail(3), use_container_width=True)

# ==========================================
# 3. LIVE INTERACTIVE DATA VIEWER CANVAS
# ==========================================
st.markdown("---")
st.subheader("🔎 Live Workspace File Inspector")

all_current_files = [
    f for f in os.listdir('.') 
    if f.endswith(SUPPORTED_EXTENSIONS) and f != "agent_tokens_logs.csv" and not f.startswith("~")
]

if all_current_files:
    view_target = st.selectbox(
        "Select file to view in full:", 
        all_current_files, 
        key="workspace_dataframe_viewer"
    )
    
    if view_target and os.path.exists(view_target):
        try:
            _, ext = os.path.splitext(view_target)
            ext = ext.lower()
            
            if ext == '.csv': full_display_df = pd.read_csv(view_target)
            elif ext == '.json': full_display_df = pd.read_json(view_target)
            else: full_display_df = pd.read_parquet(view_target)
                
            st.write(f"Showing full interactive layout for `{view_target}` ({full_display_df.shape[0]} total rows):")
            st.dataframe(full_display_df, use_container_width=True)
            
        except Exception as read_error:
            st.error(f"Could not render full dataset preview: {str(read_error)}")
else:
    st.info("No data files currently exist in the active workspace environment folder root.")