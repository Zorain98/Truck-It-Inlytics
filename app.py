import streamlit as st
import os
import shutil
from io import StringIO
import requests
import pandas as pd
from dotenv import load_dotenv
from langchain_groq.chat_models import ChatGroq
import pandasai as pai
from pandasai_openai.openai import OpenAI as PandasAI_OpenAI
import warnings
import plotly.express as px
import plotly.graph_objects as go
from datetime import datetime
import time

warnings.filterwarnings("ignore")

# Page configuration
st.set_page_config(
    page_title="Truck It Inlytics",
    page_icon="🚛",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom CSS for better UI
st.markdown("""
<style>
    .main-header {
        background: linear-gradient(90deg, #667eea 0%, #764ba2 100%);
        padding: 2rem;
        border-radius: 10px;
        margin-bottom: 2rem;
        text-align: center;
        color: white;
    }
    
    .chat-container {
        background-color: #f8f9fa;
        border-radius: 10px;
        padding: 1rem;
        margin: 1rem 0;
    }
    
    .human-message {
        background-color: #e3f2fd;
        border-radius: 15px;
        padding: 10px 15px;
        margin: 5px 0;
        border-left: 4px solid #2196f3;
    }
    
    .bot-message {
        background-color: #f3e5f5;
        border-radius: 15px;
        padding: 10px 15px;
        margin: 5px 0;
        border-left: 4px solid #9c27b0;
    }
    
    .metric-card {
        background: white;
        padding: 1rem;
        border-radius: 10px;
        box-shadow: 0 2px 4px rgba(0,0,0,0.1);
        text-align: center;
    }
    
    .stButton > button {
        background: linear-gradient(45deg, #667eea, #764ba2);
        color: white;
        border: none;
        border-radius: 25px;
        padding: 0.5rem 2rem;
        font-weight: bold;
        transition: all 0.3s;
    }
    
    .stButton > button:hover {
        transform: translateY(-2px);
        box-shadow: 0 4px 8px rgba(0,0,0,0.2);
    }
    
    .sidebar-header {
        background: linear-gradient(45deg, #667eea, #764ba2);
        color: white;
        padding: 1rem;
        border-radius: 10px;
        text-align: center;
        margin-bottom: 1rem;
    }
</style>
""", unsafe_allow_html=True)

# Initialize session state
if 'agent_initialized' not in st.session_state:
    st.session_state.agent_initialized = False
if 'smart_df' not in st.session_state:
    st.session_state.smart_df = None
if 'chat_history' not in st.session_state:
    st.session_state.chat_history = []
if 'llm' not in st.session_state:
    st.session_state.llm = None

def clean_cache():
    """Clean up cache and exports"""
    shutil.rmtree("cache", ignore_errors=True)
    shutil.rmtree("exports", ignore_errors=True)

def initialize_llm(provider, api_key, model_name):
    """Initialize LLM based on provider"""
    try:
        if provider == "OpenAI":
            llm = PandasAI_OpenAI(
                api_token=api_key,
                model=model_name,
                temperature=0.3,
            )
        else:  # Groq
            # Note: You might need to create a Groq wrapper for pandasai
            # For now, using OpenAI format - adjust as needed
            llm = PandasAI_OpenAI(
                api_token=api_key,
                model=model_name,
                temperature=0.3,
            )
        return llm
    except Exception as e:
        st.error(f"Error initializing LLM: {str(e)}")
        return None

def load_data_from_url(url):
    """Load data from Redash API URL"""
    try:
        df = pai.read_csv(url)
        return df
    except Exception as e:
        st.error(f"Error loading data from URL: {str(e)}")
        return None

def load_data_from_file(uploaded_file):
    """Load data from uploaded CSV file"""
    try:
        df = pd.read_csv(uploaded_file)
        smart_df = pai.SmartDataframe(df, config={"llm": st.session_state.llm})
        return smart_df
    except Exception as e:
        st.error(f"Error loading uploaded file: {str(e)}")
        return None

def display_data_summary(df):
    """Display data summary and statistics"""
    if df is None:
        st.warning("No data available for summary")
        return
    
    # Convert to pandas DataFrame if it's a SmartDataframe
    if hasattr(df, 'dataframe'):
        data = df.dataframe
    else:
        data = df
    
    st.markdown("### 📊 Data Overview")
    
    col1, col2, col3, col4 = st.columns(4)
    
    with col1:
        st.markdown(f"""
        <div class="metric-card">
            <h3 style="color: #667eea;">📝 Rows</h3>
            <h2>{len(data):,}</h2>
        </div>
        """, unsafe_allow_html=True)
    
    with col2:
        st.markdown(f"""
        <div class="metric-card">
            <h3 style="color: #764ba2;">📋 Columns</h3>
            <h2>{len(data.columns)}</h2>
        </div>
        """, unsafe_allow_html=True)
    
    with col3:
        numeric_cols = data.select_dtypes(include=['number']).columns
        st.markdown(f"""
        <div class="metric-card">
            <h3 style="color: #667eea;">🔢 Numeric</h3>
            <h2>{len(numeric_cols)}</h2>
        </div>
        """, unsafe_allow_html=True)
    
    with col4:
        text_cols = data.select_dtypes(include=['object']).columns
        st.markdown(f"""
        <div class="metric-card">
            <h3 style="color: #764ba2;">📝 Text</h3>
            <h2>{len(text_cols)}</h2>
        </div>
        """, unsafe_allow_html=True)
    
    st.markdown("### 🔍 Data Preview")
    st.dataframe(data.head(10), use_container_width=True)
    
    # Data types
    st.markdown("### 📋 Column Information")
    col_info = pd.DataFrame({
        'Column': data.columns,
        'Data Type': data.dtypes,
        'Non-Null Count': data.count(),
        'Null Count': data.isnull().sum()
    })
    st.dataframe(col_info, use_container_width=True)
    
    # Basic statistics for numeric columns
    if len(numeric_cols) > 0:
        st.markdown("### 📈 Numeric Statistics")
        st.dataframe(data[numeric_cols].describe(), use_container_width=True)

def main():
    # Header
    st.markdown("""
    <div class="main-header">
        <h1>🚛 Truck It Inlytics</h1>
        <p>Advanced Data Analytics Chatbot for Logistics Intelligence</p>
    </div>
    """, unsafe_allow_html=True)
    
    # Sidebar configuration
    with st.sidebar:
        st.markdown("""
        <div class="sidebar-header">
            <h2>⚙️ Configuration</h2>
        </div>
        """, unsafe_allow_html=True)
        
        # LLM Provider Selection
        st.markdown("### 🤖 Select AI Provider")
        llm_provider = st.selectbox(
            "Choose your LLM provider:",
            ["OpenAI", "Groq"],
            help="Select the AI provider for data analysis"
        )
        
        # API Key Input
        st.markdown("### 🔑 API Configuration")
        api_key = st.text_input(
            "Enter API Key:",
            type="password",
            help="Enter your API key for the selected provider"
        )
        
        # Model Selection
        if llm_provider == "OpenAI":
            model_options = ["gpt-4", "gpt-3.5-turbo", "gpt-4-turbo"]
        else:
            model_options = ["mixtral-8x7b-32768", "llama2-70b-4096"]
        
        selected_model = st.selectbox(
            "Select Model:",
            model_options,
            help="Choose the specific model to use"
        )
        
        # Initialize Agent Button
        if st.button("🚀 Initialize Agent", help="Initialize the AI agent with your settings"):
            if api_key:
                with st.spinner("Initializing agent..."):
                    clean_cache()
                    llm = initialize_llm(llm_provider, api_key, selected_model)
                    if llm:
                        st.session_state.llm = llm
                        pai.set_config({"llm": llm})
                        st.session_state.agent_initialized = True
                        st.success("✅ Agent initialized successfully!")
                    else:
                        st.error("❌ Failed to initialize agent")
            else:
                st.error("❌ Please enter your API key")
        
        st.markdown("---")
        
        # Data Source Selection
        if st.session_state.agent_initialized:
            st.markdown("### 📁 Data Source")
            data_source = st.radio(
                "Choose data source:",
                ["Upload CSV File", "Redash API URL"],
                help="Select how you want to provide data"
            )
            
            if data_source == "Upload CSV File":
                uploaded_file = st.file_uploader(
                    "Choose a CSV file",
                    type="csv",
                    help="Upload your CSV file for analysis"
                )
                
                if uploaded_file is not None:
                    with st.spinner("Loading data..."):
                        smart_df = load_data_from_file(uploaded_file)
                        if smart_df:
                            st.session_state.smart_df = smart_df
                            st.success("✅ Data loaded successfully!")
            
            else:  # Redash API URL
                redash_url = st.text_input(
                    "Enter Redash API URL:",
                    help="Enter the complete Redash API URL with authentication"
                )
                
                if st.button("📥 Load Data from URL"):
                    if redash_url:
                        with st.spinner("Loading data from URL..."):
                            smart_df = load_data_from_url(redash_url)
                            if smart_df:
                                st.session_state.smart_df = smart_df
                                st.success("✅ Data loaded successfully!")
                    else:
                        st.error("❌ Please enter a valid URL")
        
        # Clear Chat History
        if st.button("🗑️ Clear Chat History"):
            st.session_state.chat_history = []
            st.success("Chat history cleared!")

    # Main content area
    if not st.session_state.agent_initialized:
        st.info("👈 Please configure and initialize the agent in the sidebar to get started.")
        
        # Welcome information
        col1, col2 = st.columns(2)
        
        with col1:
            st.markdown("""
            ### 🌟 Features
            - **Multi-LLM Support**: Choose between OpenAI and Groq
            - **Flexible Data Input**: Upload CSV or connect to Redash
            - **Interactive Chat**: Natural language data queries
            - **Data Visualization**: Comprehensive analytics dashboard
            - **Real-time Processing**: Instant insights and responses
            """)
        
        with col2:
            st.markdown("""
            ### 🚀 Getting Started
            1. Select your preferred AI provider
            2. Enter your API key
            3. Initialize the agent
            4. Upload data or provide Redash URL
            5. Start chatting with your data!
            """)
    
    elif st.session_state.smart_df is None:
        st.info("📁 Please provide data source in the sidebar to start analysis.")
    
    else:
        # Create tabs for different functionalities
        tab1, tab2 = st.tabs(["💬 Chat with Data", "📊 Data Summary"])
        
        with tab1:
            st.markdown("### 💬 Ask Questions About Your Data")
            
            # Display chat history
            for i, (query, response) in enumerate(st.session_state.chat_history):
                st.markdown(f"""
                <div class="human-message">
                    <strong>👤 You:</strong> {query}
                </div>
                """, unsafe_allow_html=True)
                
                st.markdown(f"""
                <div class="bot-message">
                    <strong>🤖 Truck It Inlytics:</strong> {response}
                </div>
                """, unsafe_allow_html=True)
            
            # Chat input
            with st.form("chat_form", clear_on_submit=True):
                user_query = st.text_input(
                    "Ask a question about your data:",
                    placeholder="e.g., What are the top 5 pickup warehouses by total time spent?",
                    help="Type your question in natural language"
                )
                
                col1, col2 = st.columns([1, 4])
                with col1:
                    submit_button = st.form_submit_button("🚀 Ask")
                
                if submit_button and user_query:
                    with st.spinner("🤔 Analyzing your data..."):
                        try:
                            response = st.session_state.smart_df.chat(user_query)
                            
                            # Add to chat history
                            st.session_state.chat_history.append((user_query, response))
                            
                            # Display the new response
                            st.markdown(f"""
                            <div class="human-message">
                                <strong>👤 You:</strong> {user_query}
                            </div>
                            """, unsafe_allow_html=True)
                            
                            st.markdown(f"""
                            <div class="bot-message">
                                <strong>🤖 Truck It Inlytics:</strong> {response}
                            </div>
                            """, unsafe_allow_html=True)
                            
                        except Exception as e:
                            st.error(f"❌ Error processing query: {str(e)}")
        
        with tab2:
            display_data_summary(st.session_state.smart_df)

    # Footer
    st.markdown("---")
    st.markdown("""
    <div style="text-align: center; color: #666; padding: 1rem;">
        <p>🚛 Truck It Inlytics - Powered by AI for Logistics Intelligence</p>
    </div>
    """, unsafe_allow_html=True)

if __name__ == "__main__":
    main()
