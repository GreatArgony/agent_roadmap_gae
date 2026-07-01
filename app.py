import os
import streamlit as st
import pandas as pd
from langchain_core.messages import HumanMessage, AIMessage


# Pull operational variables from your background framework model file
from agent_backend import (
    director_agent, 
    director_agent_fallback, 
    config, 
    validate_user_input, 
    log_tokens, 
    CSV_LOG_FILE
)

st.set_page_config(layout="wide", page_title="AI Data Analytics Studio")
st.title("💼 Multi-Agent Workspace with Dynamic Data Ingestion")
st.markdown("---")

# Setup state persistence variables
if "messages" not in st.session_state:
    st.session_state.messages = []
if "failover_triggered" not in st.session_state:
    st.session_state.failover_triggered = False

# ==========================================
# FILE INGESTION SIDEBAR CONTROL PANEL
# ==========================================
with st.sidebar:
    st.header("📂 Data Ingestion Controller")
    
    # Mode Toggle: Upload New vs Select Existing Local Dataset File
    source_mode = st.radio("Select Data Input Source Mode:", ["Choose Local Workspace File", "Upload New File Asset"])
    
    active_file_path = None
    
    if source_mode == "Choose Local Workspace File":
        # Scan folder directory and filter out structured data profiles extensions
        supported_extensions = ('.csv', '.xlsx', '.json', '.parquet', '.xml', '.sql')
        local_files = [f for f in os.listdir('.') if f.endswith(supported_extensions)]
        
        if local_files:
            selected_filename = st.selectbox("Select target active dataset file:", local_files)
            active_file_path = selected_filename
            st.success(f"Selected Target: `{active_file_path}`")
        else:
            st.warning("No standard data frames detected inside local workspace directories.")
            
    else:
        uploaded_file = st.file_uploader("Drag and drop your file here:", type=["csv", "xlsx", "json", "parquet", "xml"])
        if uploaded_file is not None:
            # Secure write stream block to save asset to the execution directory root
            active_file_path = uploaded_file.name
            with open(active_file_path, "wb") as f:
                f.write(uploaded_file.getbuffer())
            st.success(f"Uploaded and staged: `{active_file_path}`")

    # Display a mini preview of the selected data inside the sidebar panel
    if active_file_path and os.path.exists(active_file_path):
        st.markdown("---")
        st.subheader("📊 Instant Spreadsheet Preview")
        try:
            _, ext = os.path.splitext(active_file_path)
            if ext.lower() == '.csv':
                preview_df = pd.read_csv(active_file_path, nrows=5)
            elif ext.lower() == '.xlsx':
                preview_df = pd.read_excel(active_file_path, engine='openpyxl', nrows=5)
            elif ext.lower() == '.json':
                preview_df = pd.read_json(active_file_path).head(5)
            else:
                preview_df = None
                
            if preview_df is not None:
                st.dataframe(preview_df, use_container_width=True)
        except Exception as preview_error:
            st.caption(f"Preview rendering skipped: {str(preview_error)}")

# ==========================================
# MAIN INTERFACE TWO COLUMN LAYOUT MATRIX
# ==========================================
terminal_col, graphic_col = st.columns([1, 1])

with terminal_col:
    st.subheader("Supervisor Console Interface")
    
    # Render historical conversation logs
    for interaction in st.session_state.messages:
        with st.chat_message(interaction["role"]):
            st.markdown(interaction["content"])

    # Human query text capture loop
    # Open app.py, locate your chat input block, and replace it with this:

if user_prompt := st.chat_input("Enter command parameters..."):
    if not active_file_path:
        st.warning("⚠️ Access Denied: Please select or upload a dataset file in the sidebar panel before executing instructions.")
    else:
        contextualized_prompt = f"Using the dataset file '{active_file_path}', please execute the following instruction: {user_prompt}"
        
        st.chat_message("user").write(user_prompt)
        st.session_state.messages.append({"role": "user", "content": user_prompt})
        
        is_safe, error_message = validate_user_input(user_prompt, max_token_count=100)
        
        if not is_safe:
            st.error(error_message)
            st.session_state.messages.append({"role": "assistant", "content": f"Halted: {error_message}"})
        else:
            with st.chat_message("assistant"):
                status_loader = st.empty()
                status_loader.info(f"Director is guiding tasks using target asset context: `{active_file_path}`...")
                
                # --- UPGRADE: Set a high recursion limit for our complex multi-agent framework ---
                runtime_config = {
                    "configurable": config["configurable"],
                    "recursion_limit": 100  # Gives sub-agents plenty of room to pass steps
                }
                
                try:
                    payload = {"messages": [HumanMessage(content=contextualized_prompt)]}
                    
                    # Run the normal agent with the high step limit
                    if st.session_state.failover_triggered:
                        execution_output = director_agent_fallback.invoke(payload, config=runtime_config)
                    else:
                        execution_output = director_agent.invoke(payload, config=runtime_config)
                    
                    final_reply = execution_output["messages"][-1]
                    status_loader.empty()
                    st.markdown(final_reply.content)
                    st.session_state.messages.append({"role": "assistant", "content": final_reply.content})
                    
                    if isinstance(final_reply, AIMessage):
                        log_tokens(final_reply)
                        
                except Exception as api_outage:
                    # --- UPGRADE: Clean failover execution path without recursive catch loops ---
                    status_loader.warning(f"Primary engine encountered an error: {str(api_outage)}. Deploying premium fallback model routing...")
                    
                    if not st.session_state.failover_triggered:
                        st.session_state.failover_triggered = True
                        try:
                            # Run fallback agent with the proper steps configuration
                            execution_output = director_agent_fallback.invoke(payload, config=runtime_config)
                            final_reply = execution_output["messages"][-1]
                            status_loader.empty()
                            st.markdown(final_reply.content)
                            st.session_state.messages.append({"role": "assistant", "content": final_reply.content})
                            
                            if isinstance(final_reply, AIMessage):
                                log_tokens(final_reply)
                        except Exception as fallback_error:
                            status_loader.empty()
                            error_details = f"Both primary and fallback engines failed to parse this instruction. Error trace: {str(fallback_error)}"
                            st.error(error_details)
                            st.session_state.messages.append({"role": "assistant", "content": error_details})
                    else:
                        status_loader.empty()
                        error_details = f"System recovery failed. Underlying structural exception: {str(api_outage)}"
                        st.error(error_details)
                        st.session_state.messages.append({"role": "assistant", "content": error_details})
                        
            st.rerun()

with graphic_col:
    st.subheader("Data Visualization Display Output")
    
    if os.path.exists("data_analyzer.png"):
        st.image("data_analyzer.png", use_container_width=True, caption="Pipeline Chart Output Canvas Container")
        if st.button("Flush Visual Assets Cache"):
            os.remove("data_analyzer.png")
            st.rerun()
    else:
        st.info("Awaiting structural chart generation operations on current transaction execution streams.")
        
    st.markdown("---")
    st.subheader("Telemetry Log Audit Ledger (CSV Overview)")
    
    if os.path.exists(CSV_LOG_FILE):
        log_df = pd.read_csv(CSV_LOG_FILE)
        st.dataframe(log_df.tail(5), use_container_width=True)
    else:
        st.caption("Awaiting initial system telemetry transactions...")