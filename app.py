import streamlit as st
import pandas as pd
import numpy as np
import os
from datetime import datetime
import plotly.express as px
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import requests
import json
from langchain_openai import ChatOpenAI
from langchain.prompts import ChatPromptTemplate
from langchain_core.messages import SystemMessage, HumanMessage

# Page config
st.set_page_config(
    page_title="Truck It Inlytics",
    page_icon="🚛",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom CSS for better aesthetics
st.markdown("""
<style>
    .main {
        padding-top: 2rem;
    }
    
    .stChat {
        background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
        border-radius: 10px;
        padding: 20px;
        margin: 10px 0;
    }
    
    .metric-card {
        background: linear-gradient(135deg, #000046 0%, #1CB5E0 100%);
        padding: 20px;
        border-radius: 15px;
        color: white;
        text-align: center;
        box-shadow: 0 4px 15px rgba(0,0,0,0.1);
        margin: 10px 0;
    }
    
    .metric-value {
        font-size: 2.5rem;
        font-weight: bold;
        margin: 10px 0;
    }
    
    .metric-label {
        font-size: 1.1rem;
        opacity: 0.9;
    }
    
    .sidebar .sidebar-content {
        background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
    }
    
    .stButton > button {
        background: linear-gradient(135deg, #000046 0%, #1CB5E0 100%);
        color: white;
        border: none;
        border-radius: 25px;
        padding: 0.5rem 2rem;
        font-weight: 600;
        transition: all 0.3s ease;
        box-shadow: 0 4px 15px rgba(0,0,0,0.2);
    }
    
    .stButton > button:hover {
        color: white;
        transform: translateY(-2px);
        box-shadow: 0 6px 20px rgba(0,0,0,0.3);
    }
    
    .upload-section {
        background: linear-gradient(135deg, #000046 0%, #1CB5E0 100%);
        color: white;
        padding: 20px;
        border-radius: 15px;
        margin: 20px 0;
    }
    
    .chat-message {
        padding: 15px;
        border-radius: 15px;
        margin: 10px 0;
        display: flex;
        align-items: flex-start;
        gap: 10px;
    }
    
    .human-message {
        background: linear-gradient(135deg, #127555 0%, #099773 100%);
        color: white;
        margin-left: 20px;
    }
    
    .agent-message {
        background: linear-gradient(135deg, #000438 0%, #00458E 100%);
        color: white;
        margin-right: 20px;
    }
    
    .avatar {
        width: 40px;
        height: 40px;
        border-radius: 50%;
        display: flex;
        align-items: center;
        justify-content: center;
        font-size: 20px;
        font-weight: bold;
    }
    
    .human-avatar {
        background: white;
        color: white;
    }
    
    .agent-avatar {
        background: white;
        color: white;
    }
</style>
""", unsafe_allow_html=True)

# Initialize session state
if 'messages' not in st.session_state:
    st.session_state.messages = []
if 'agent_initialized' not in st.session_state:
    st.session_state.agent_initialized = False
if 'df' not in st.session_state:
    st.session_state.df = None
if 'llm_type' not in st.session_state:
    st.session_state.llm_type = None
if 'api_key' not in st.session_state:
    st.session_state.api_key = None

def initialize_pandasai():
    """Initialize PandasAI with the selected LLM"""
    try:
        import pandasai as pai
        if st.session_state.llm_type == "OpenAI":
            from pandasai_openai.openai import OpenAI
            llm = OpenAI(api_token=st.session_state.api_key)
        elif st.session_state.llm_type == "Groq":
            # Use PandasAI's built-in LLM wrapper for external APIs
            from pandasai.llm import LLM
            
            class GroqLLM(LLM):
                def __init__(self, api_key):
                    self.api_key = api_key
                    self._client = None
                
                @property
                def client(self):
                    if self._client is None:
                        import groq
                        self._client = groq.Groq(api_key=self.api_key)
                    return self._client
                
                def call(self, instruction, value):
                    try:
                        response = self.client.chat.completions.create(
                            model="llama-3.3-70b-versatile",
                            messages=[
                                {"role": "system", "content": instruction},
                                {"role": "user", "content": value}
                            ],
                            temperature=0.1
                        )
                        return response.choices[0].message.content
                    except Exception as e:
                        return f"Error: {str(e)}"
            
            llm = GroqLLM(st.session_state.api_key)
        
        pai.config.set({"llm": llm})
        st.session_state.agent_initialized = True
        return True
    except Exception as e:
        st.error(f"Error initializing agent: {str(e)}")
        return False
def load_data_from_redash(api_url):
    """Load data from Redash API URL using PandasAI"""
    try:
        import pandasai as pai
        # Store the user given Redash API url in a variable
        redash_api_url = api_url
        
        # Use PandasAI to read CSV directly from the URL
        df = pai.read_csv(redash_api_url)
        
        return df
    except Exception as e:
        st.error(f"Error loading data from Redash: {str(e)}")
        return None


def display_data_summary(df):
    """Display comprehensive data summary and statistics using PandasAI"""
    if df is None:
        return
    
    st.markdown("## 📊 Data Summary & Statistics")
    
    # Get the underlying pandas DataFrame for analysis
    pandas_df = df.dataframe if hasattr(df, 'dataframe') else df
    
    # Basic metrics in cards
    col1, col2, col3, col4 = st.columns(4)
    
    with col1:
        st.markdown(f"""
        <div class="metric-card">
            <div class="metric-label">Total Records</div>
            <div class="metric-value">{len(pandas_df):,}</div>
        </div>
        """, unsafe_allow_html=True)
    
    with col2:
        st.markdown(f"""
        <div class="metric-card">
            <div class="metric-label">Total Columns</div>
            <div class="metric-value">{len(pandas_df.columns)}</div>
        </div>
        """, unsafe_allow_html=True)
    
    with col3:
        numeric_cols = pandas_df.select_dtypes(include=[np.number]).columns
        st.markdown(f"""
        <div class="metric-card">
            <div class="metric-label">Numeric Columns</div>
            <div class="metric-value">{len(numeric_cols)}</div>
        </div>
        """, unsafe_allow_html=True)
    
    with col4:
        missing_percentage = (pandas_df.isnull().sum().sum() / (len(pandas_df) * len(pandas_df.columns))) * 100
        st.markdown(f"""
        <div class="metric-card">
            <div class="metric-label">Missing Data %</div>
            <div class="metric-value">{missing_percentage:.1f}%</div>
        </div>
        """, unsafe_allow_html=True)
    
    # Data overview tabs
    tab1, tab2, tab3, tab4 = st.tabs(["📋 Data Preview", "📈 Statistics", "🔍 Data Types", "📊 Visualizations"])
    
    with tab1:
        st.markdown("### Data Preview")
        st.dataframe(pandas_df.head(20), use_container_width=True)
        
        st.markdown("### Data Shape")
        st.info(f"Dataset contains **{len(pandas_df)} rows** and **{len(pandas_df.columns)} columns**")
    
    with tab2:
        st.markdown("### Statistical Summary")
        if len(numeric_cols) > 0:
            st.dataframe(pandas_df[numeric_cols].describe(), use_container_width=True)
        else:
            st.info("No numeric columns found for statistical analysis")
        
        st.markdown("### Missing Values Analysis")
        missing_data = pandas_df.isnull().sum()
        missing_data = missing_data[missing_data > 0].sort_values(ascending=False)
        
        if len(missing_data) > 0:
            # Convert to Python native types to avoid JSON serialization issues
            missing_values = [int(x) for x in missing_data.values]
            missing_columns = [str(x) for x in missing_data.index]
            
            fig = px.bar(
                x=missing_values,
                y=missing_columns,
                orientation='h',
                title="Missing Values by Column",
                color=missing_values,
                color_continuous_scale="Reds"
            )
            fig.update_layout(height=400)
            st.plotly_chart(fig, use_container_width=True)
        else:
            st.success("🎉 No missing values found in the dataset!")
    
    with tab3:
        st.markdown("### Data Types Analysis")
        dtype_counts = pandas_df.dtypes.value_counts()
        
        col1, col2 = st.columns(2)
        
        with col1:
            # Convert dtype names to strings to avoid JSON serialization issues
            dtype_names = [str(dtype) for dtype in dtype_counts.index]
            dtype_values = [int(count) for count in dtype_counts.values]
            
            fig = px.pie(
                values=dtype_values,
                names=dtype_names,
                title="Distribution of Data Types"
            )
            st.plotly_chart(fig, use_container_width=True)
        
        with col2:
            import pandas as pd
            dtype_df = pd.DataFrame({
                'Column': pandas_df.columns,
                'Data Type': pandas_df.dtypes.astype(str),
                'Non-Null Count': pandas_df.count(),
                'Null Count': pandas_df.isnull().sum()
            })
            st.dataframe(dtype_df, use_container_width=True)
    
    with tab4:
        st.markdown("### Data Visualizations")
        
        if len(numeric_cols) > 0:
            # Correlation heatmap
            if len(numeric_cols) > 1:
                st.markdown("#### Correlation Matrix")
                corr_matrix = pandas_df[numeric_cols].corr()
                # Convert column names to strings
                corr_matrix.columns = [str(col) for col in corr_matrix.columns]
                corr_matrix.index = [str(idx) for idx in corr_matrix.index]
                
                fig = px.imshow(
                    corr_matrix,
                    title="Correlation Heatmap",
                    color_continuous_scale="RdBu",
                    aspect="auto"
                )
                st.plotly_chart(fig, use_container_width=True)
            
            # Distribution plots
            if len(numeric_cols) > 0:
                st.markdown("#### Distribution Analysis")
                # Convert numeric columns to strings for selectbox
                numeric_cols_str = [str(col) for col in numeric_cols]
                selected_col = st.selectbox("Select column for distribution analysis:", numeric_cols_str)
                
                if selected_col:
                    col1, col2 = st.columns(2)
                    
                    with col1:
                        fig = px.histogram(
                            pandas_df,
                            x=selected_col,
                            title=f"Distribution of {selected_col}",
                            nbins=30
                        )
                        st.plotly_chart(fig, use_container_width=True)
                    
                    with col2:
                        fig = px.box(
                            pandas_df,
                            y=selected_col,
                            title=f"Box Plot of {selected_col}"
                        )
                        st.plotly_chart(fig, use_container_width=True)

def handle_pandasai_response(result, query, smart_df):
    """Handle PandasAI responses and format them properly"""
    try:
        import pandas as pd
        
        # Get the underlying pandas DataFrame
        if hasattr(smart_df, 'dataframe'):
            df = smart_df.dataframe
        else:
            df = smart_df
        
        # Analyze the query to determine what kind of response to give
        query_lower = query.lower()
        
        # If asking to show/display/list data
        if any(word in query_lower for word in ['show', 'display', 'list', 'all', 'first', 'top']):
            # Extract number if specified (like "show first 5", "top 10")
            import re
            numbers = re.findall(r'\d+', query)
            limit = int(numbers[0]) if numbers else 10
            
            # Get the data to display
            display_df = df.head(limit)
            
            return create_styled_table(display_df, f"Showing {len(display_df)} records")
        
        # If asking for count/total/number
        elif any(word in query_lower for word in ['count', 'total', 'number', 'how many']):
            count = len(df)
            return create_metric_card(f"{count:,}", "Total Records", "📊")
        
        # If asking for specific columns or filtering
        elif any(word in query_lower for word in ['rider', 'name', 'id']):
            # Try to show relevant columns
            relevant_cols = []
            if 'rider' in query_lower or 'id' in query_lower:
                relevant_cols.extend([col for col in df.columns if 'id' in col.lower() or 'rider' in col.lower()])
            if 'name' in query_lower:
                relevant_cols.extend([col for col in df.columns if 'name' in col.lower()])
            
            if relevant_cols:
                # Remove duplicates while preserving order
                relevant_cols = list(dict.fromkeys(relevant_cols))
                display_df = df[relevant_cols].head(10)
                return create_styled_table(display_df, f"Showing {relevant_cols} data")
        
        # If the result looks like it should be a table but came as string
        if isinstance(result, str) and any(char.isdigit() for char in result):
            # Try to parse and create a proper table
            parsed_df = parse_result_to_dataframe(result, df)
            if parsed_df is not None:
                return create_styled_table(parsed_df, "Query Results")
        
        # For simple answers (numbers, short text)
        if isinstance(result, (int, float)):
            return create_metric_card(f"{result:,.2f}", query, "🔢")
        elif isinstance(result, str) and len(result.split()) <= 20:
            return create_text_answer(result, query)
        
        # Default: return as formatted text
        return create_text_answer(str(result), query)
        
    except Exception as e:
        return f"<div style='color: red;'>Error processing response: {str(e)}</div>"

def parse_result_to_dataframe(result_string, original_df):
    """Try to parse string result back to DataFrame"""
    try:
        import pandas as pd
        import re
        
        # If the string contains column names from original DataFrame
        lines = result_string.strip().split('\n')
        data = []
        
        for line in lines:
            if line.strip():
                # Try to parse each line
                parts = re.split(r'\s+', line.strip())
                if len(parts) >= 2:
                    try:
                        # Assuming first part is index, second is numeric, rest is text
                        index_val = parts[0]
                        if index_val.isdigit():
                            row_data = {}
                            
                            # Map to original DataFrame columns if possible
                            if len(parts) >= 3:
                                row_data[original_df.columns[0]] = parts[1] if len(original_df.columns) > 0 else parts[1]
                                if len(original_df.columns) > 1:
                                    row_data[original_df.columns[1]] = ' '.join(parts[2:])
                            
                            if row_data:
                                data.append(row_data)
                    except (ValueError, IndexError):
                        continue
        
        if data:
            return pd.DataFrame(data)
        return None
        
    except Exception:
        return None

def create_styled_table(df, title="Results"):
    """Create a beautifully styled HTML table"""
    if df.empty:
        return f"<div style='text-align: center; color: #666;'>No data to display</div>"
    
    # Create HTML table with custom styling
    html_table = df.to_html(
        index=False,
        classes='result-table',
        escape=False,
        table_id='data-table'
    )
    
    return f"""
    <div style="margin: 15px 0;">
        <h4 style="color: #667eea; margin-bottom: 10px; font-weight: 600;">{title}</h4>
        <style>
            .result-table {{
                border-collapse: collapse;
                width: 100%;
                margin: 15px 0;
                font-size: 14px;
                border-radius: 8px;
                overflow: hidden;
                box-shadow: 0 4px 6px rgba(0, 0, 0, 0.1);
            }}
            .result-table thead tr {{
                background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
                color: white;
                text-align: left;
                font-weight: 600;
            }}
            .result-table th,
            .result-table td {{
                padding: 12px 15px;
                border: none;
                text-align: left;
            }}
            .result-table tbody tr {{
                border-bottom: 1px solid #f0f0f0;
                transition: background-color 0.3s ease;
            }}
            .result-table tbody tr:nth-of-type(even) {{
                background-color: #f8f9fa;
            }}
            .result-table tbody tr:hover {{
                background-color: #e3f2fd;
                transform: translateY(-1px);
                box-shadow: 0 2px 4px rgba(0,0,0,0.1);
            }}
        </style>
        {html_table}
        <small style="color: #666; font-style: italic;">Showing {len(df)} row(s)</small>
    </div>
    """

def create_metric_card(value, label, icon="📊"):
    """Create a metric card for numerical results"""
    return f"""
    <div style="background: linear-gradient(135deg, #667eea 0%, #764ba2 100%); 
                color: white; padding: 20px; border-radius: 12px; 
                text-align: center; margin: 15px 0; box-shadow: 0 4px 15px rgba(0,0,0,0.2);">
        <div style="font-size: 2.5em; margin-bottom: 5px;">{icon}</div>
        <div style="font-size: 2.2em; font-weight: bold; margin: 10px 0;">{value}</div>
        <div style="font-size: 1.1em; opacity: 0.9;">{label}</div>
    </div>
    """

def create_text_answer(text, query):
    """Create a formatted text answer"""
    return f"""
    <div style="background: linear-gradient(135deg, #f093fb 0%, #f5576c 100%); 
                color: white; padding: 20px; border-radius: 12px; margin: 15px 0;
                box-shadow: 0 4px 15px rgba(0,0,0,0.2);">
        <div style="font-weight: 600; margin-bottom: 10px; font-size: 1.1em;">Response:</div>
        <div style="font-size: 1.1em;">{text}</div>
    </div>
    """


def main():
    # Header
    st.markdown("""
    <div style="text-align: center; padding: 2rem 0;">
        <h1 style="color: #667eea; font-size: 3rem; margin-bottom: 0;">🚛 Truck It Inlytics</h1>
        <p style="color: #000000; font-size: 1.2rem;">AI-Powered Data Analytics Chatbot</p>
    </div>
    """, unsafe_allow_html=True)
    
    # Sidebar configuration
    with st.sidebar:
        st.markdown("## ⚙️ Configuration")
        
        # LLM Selection
        llm_choice = st.selectbox(
            "Select LLM Provider:",
            ["Select...", "OpenAI", "Groq"],
            key="llm_selector"
        )
        
        if llm_choice != "Select...":
            st.session_state.llm_type = llm_choice
            
            # API Key input
            api_key_label = f"{llm_choice} API Key:"
            api_key = st.text_input(
                api_key_label,
                type="password",
                placeholder=f"Enter your {llm_choice} API key..."
            )
            
            if api_key:
                st.session_state.api_key = api_key
                
                # Initialize Agent button
                if st.button("🚀 Initialize Agent", key="init_agent"):
                    with st.spinner("Initializing agent..."):
                        if initialize_pandasai():
                            st.success("✅ Agent initialized successfully!")
                        else:
                            st.error("❌ Failed to initialize agent")
        
        st.markdown("---")
        
        # Data source selection
        if st.session_state.agent_initialized:
            st.markdown("## 📁 Data Source")
            
            data_source = st.radio(
                "Choose data source:",
                ["Upload CSV", "Redash API URL"],
                key="data_source"
            )
            
            if data_source == "Upload CSV":
                uploaded_file = st.file_uploader(
                    "Upload CSV file:",
                    type="csv",
                    help="Upload a CSV file to analyze"
                )
                
                if uploaded_file is not None:
                    try:
                        import pandasai as pai
                        df = pai.read_csv(uploaded_file)
                        st.session_state.df = df
                        st.success(f"✅ File uploaded! {len(df)} rows loaded.")
                    except Exception as e:
                        st.error(f"Error loading file: {str(e)}")
            
            elif data_source == "Redash API URL":
                redash_url = st.text_input(
                    "Redash API URL:",
                    placeholder="Enter Redash API URL..."
                )
                
                if redash_url and st.button("📥 Load Data"):
                    with st.spinner("Loading data from Redash..."):
                        df = load_data_from_redash(redash_url)
                        if df is not None:
                            st.session_state.df = df
                            st.success(f"✅ Data loaded! {len(df)} rows loaded.")
        
        # Display current status
        st.markdown("---")
        st.markdown("## 📊 Status")
        
        agent_status = "✅ Ready" if st.session_state.agent_initialized else "❌ Not initialized"
        data_status = "✅ Loaded" if st.session_state.df is not None else "❌ No data"
        
        st.markdown(f"**Agent:** {agent_status}")
        st.markdown(f"**Data:** {data_status}")
        
        if st.session_state.df is not None:
            st.markdown(f"**Rows:** {len(st.session_state.df):,}")
            st.markdown(f"**Columns:** {len(st.session_state.df.columns)}")

    # Main content area
    if not st.session_state.agent_initialized:
        st.markdown("""
        <div class="upload-section">
            <h3>🚀 Getting Started</h3>
            <p>To begin using Truck It Inlytics, please:</p>
            <ol>
                <li>Select your preferred LLM provider from the sidebar</li>
                <li>Enter your API key</li>
                <li>Click "Initialize Agent"</li>
                <li>Upload a CSV file or provide a Redash API URL</li>
            </ol>
        </div>
        """, unsafe_allow_html=True)
        
        # Demo features
        st.markdown("## ✨ Features")
        
        col1, col2, col3 = st.columns(3)
        
        with col1:
            st.markdown("""
            ### 🤖 AI Chat Interface
            - Natural language queries
            - Intelligent data analysis
            - Real-time responses
            """)
        
        with col2:
            st.markdown("""
            ### 📊 Data Visualization
            - Interactive charts
            - Statistical summaries
            - Data insights
            """)
        
        with col3:
            st.markdown("""
            ### 🔧 Multiple Data Sources
            - CSV file upload
            - Redash API integration
            - Real-time data loading
            """)
    
    else:
        # Create tabs for different interfaces
        if st.session_state.df is not None:
            tab1, tab2 = st.tabs(["💬 Chat Interface", "📊 Data Summary"])
            
            with tab1:
                st.markdown("## 💬 Chat with Your Data")
                
                # Display chat messages
                for message in st.session_state.messages:
                    if message["role"] == "user":
                        st.markdown(f"""
                        <div class="chat-message human-message">
                            <div class="avatar human-avatar">👤</div>
                            <div>{message["content"]}</div>
                        </div>
                        """, unsafe_allow_html=True)
                    else:
                        st.markdown(f"""
                        <div class="chat-message agent-message">
                            <div class="avatar agent-avatar">👾</div>
                            <div>{message["content"]}</div>
                        </div>
                        """, unsafe_allow_html=True)
                
                # Chat input
                if query := st.chat_input("Ask me anything about your data..."):
                    # Add user message
                    st.session_state.messages.append({"role": "user", "content": query})
                    
                    # Process query with PandasAI
                    try:
                        import pandasai as pai
                        with st.spinner("🤖 Analyzing your data..."):
                            # Ensure we have a SmartDataframe
                            if not hasattr(st.session_state.df, 'chat'):
                                if hasattr(st.session_state.df, 'dataframe'):
                                    pandas_df = st.session_state.df.dataframe
                                else:
                                    pandas_df = st.session_state.df
                                st.session_state.df = pai.SmartDataframe(pandas_df)
                            
                            # Get response from PandasAI
                            result = st.session_state.df.chat(query)
                            
                            # Instead of using the result directly, let's try to execute the query manually
                            formatted_response = handle_pandasai_response(result, query, st.session_state.df)
                            
                            # Add agent response
                            st.session_state.messages.append({"role": "assistant", "content": formatted_response})
                            
                            # Rerun to update chat display
                            st.rerun()
                    
                    except Exception as e:
                        error_msg = f"Sorry, I encountered an error: {str(e)}"
                        st.session_state.messages.append({"role": "assistant", "content": error_msg})
                        st.rerun()

            
            with tab2:
                display_data_summary(st.session_state.df)
        
        else:
            st.markdown("""
            <div class="upload-section">
                <h3>📁 No Data Loaded</h3>
                <p>Please upload a CSV file or provide a Redash API URL from the sidebar to start analyzing your data.</p>
            </div>
            """, unsafe_allow_html=True)

if __name__ == "__main__":
    main()
