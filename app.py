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
import re

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

def reformat_output_with_llm(raw_response, user_query, openai_api_key):
    # Use the OpenAI model from LangChain to turn string output into a table or concise summary.
    prompt = ChatPromptTemplate.from_messages(
        [
            SystemMessage(content=(
                "You are a data assistant. "
                "When given a raw text output (possibly with table data), "
                "format it into a readable Markdown table or, if a table is not relevant, "
                "give a concise, well-organized summary. "
                "Be accurate and relevant to the input query."
            )),
            HumanMessage(content=f"User Query: {user_query}\nRaw Output: {raw_response}\n\n---\nReturn ONLY a Table (Markdown) or concise answer. If table is too wide, include only meaningful columns.")
        ]
    )
    llm = ChatOpenAI(
        model="gpt-3.5-turbo",
        api_key=openai_api_key,
        temperature=0.1,
        # Optionally, set max_tokens=512 or similar
    )
    response = llm.invoke(prompt)
    return response.content

def try_show_dataframe(query, df):
    # Lower-case query for easy matching
    query_lower = query.lower()
    
    # Try basic table requests
    keywords = ["show", "list", "table", "display", "top", "head"]
    if any(word in query_lower for word in keywords):
        # Try to extract relevant columns from the query
        possible_cols = []
        for col in df.columns:
            if col.lower() in query_lower:
                possible_cols.append(col)
        if possible_cols:
            display_df = df[possible_cols].head(20)
        else:
            display_df = df.head(20)
        return display_df
    return None

def smart_query_router(query, df):
    """
    Routes queries to appropriate handlers based on intent detection.
    Returns (handled, result) where handled is True if we processed it ourselves.
    """
    query_lower = query.lower()
    
    # Extract numbers from query for limits
    numbers = re.findall(r'\d+', query)
    limit = int(numbers[0]) if numbers else 10
    
    # 1. Show/Display/List Queries
    if any(word in query_lower for word in ['show', 'display', 'list', 'all', 'first', 'top', 'head']):
        # Check for specific columns mentioned
        relevant_cols = get_relevant_columns(query_lower, df)
        if relevant_cols:
            result_df = df[relevant_cols].head(limit)
            return True, create_smart_table(result_df, f"Showing {limit} {' & '.join(relevant_cols)} records")
        else:
            result_df = df.head(limit)
            return True, create_smart_table(result_df, f"Showing first {limit} records")
    
    # 2. Count/Total Queries
    elif any(word in query_lower for word in ['count', 'total', 'how many', 'number of']):
        total = len(df)
        return True, create_count_card(total, "Total Records")
    
    # 3. Search/Filter Queries
    elif any(word in query_lower for word in ['find', 'search', 'where', 'filter', 'contains']):
        filtered_df = smart_filter(query, df)
        if not filtered_df.empty:
            return True, create_smart_table(filtered_df.head(limit), f"Search Results ({len(filtered_df)} found)")
        else:
            return True, create_text_response("No matching records found.")
    
    # 4. Column Information Queries
    elif any(word in query_lower for word in ['columns', 'fields', 'what data', 'structure']):
        col_info = create_column_info(df)
        return True, col_info
    
    # If we can't handle it, return False to let PandasAI handle it
    return False, None

def get_relevant_columns(query, df):
    """Find columns that are mentioned in the query."""
    relevant_cols = []
    for col in df.columns:
        if col.lower() in query or any(word in col.lower() for word in query.split()):
            relevant_cols.append(col)
    return relevant_cols[:5]  # Limit to 5 columns max

def smart_filter(query, df):
    """Attempt to filter DataFrame based on query content."""
    query_lower = query.lower()
    
    # Extract quoted strings or specific values
    quoted_terms = re.findall(r'"([^"]*)"', query) + re.findall(r"'([^']*)'", query)
    
    if quoted_terms:
        # Filter by any column containing the quoted term
        mask = pd.Series([False] * len(df))
        for term in quoted_terms:
            for col in df.select_dtypes(include=['object']).columns:
                mask |= df[col].astype(str).str.contains(term, case=False, na=False)
        return df[mask]
    
    return pd.DataFrame()  # Return empty if no filtering possible

def create_smart_table(df, title):
    """Create a beautiful, responsive table."""
    if df.empty:
        return f"<div class='no-data'>No data to display for: {title}</div>"
    
    # Limit columns if too many
    if len(df.columns) > 6:
        display_df = df.iloc[:, :6]
        title += f" (showing first 6 of {len(df.columns)} columns)"
    else:
        display_df = df
    
    table_html = display_df.to_html(index=False, classes='smart-table', escape=False)
    
    return f"""
    <div class="table-container">
        <h4 class="table-title">{title}</h4>
        <style>
            .table-container {{
                margin: 20px 0;
                border-radius: 12px;
                overflow: hidden;
                box-shadow: 0 4px 6px rgba(0, 0, 0, 0.1);
            }}
            .table-title {{
                background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
                color: white;
                margin: 0;
                padding: 15px 20px;
                font-size: 1.1em;
                font-weight: 600;
            }}
            .smart-table {{
                width: 100%;
                border-collapse: collapse;
                font-size: 0.9em;
                background: white;
                margin: 0;
            }}
            .smart-table th {{
                background: #f8f9fa;
                color: #495057;
                font-weight: 600;
                padding: 12px 15px;
                text-align: left;
                border-bottom: 2px solid #dee2e6;
            }}
            .smart-table td {{
                padding: 10px 15px;
                border-bottom: 1px solid #dee2e6;
                color: #212529;
            }}
            .smart-table tbody tr:hover {{
                background-color: #f1f3f4;
                transition: background-color 0.2s ease;
            }}
            .smart-table tbody tr:nth-child(even) {{
                background-color: #f8f9fa;
            }}
            .no-data {{
                text-align: center;
                padding: 40px;
                color: #6c757d;
                font-style: italic;
            }}
        </style>
        {table_html}
        <div style="background: #f8f9fa; padding: 8px 15px; font-size: 0.85em; color: #6c757d; border-top: 1px solid #dee2e6;">
            📊 Showing {len(display_df)} rows
        </div>
    </div>
    """

def create_count_card(count, label):
    """Create a metric card for counts."""
    return f"""
    <div style="background: linear-gradient(135deg, #28a745 0%, #20c997 100%); 
                color: white; padding: 30px; border-radius: 15px; 
                text-align: center; margin: 20px 0; box-shadow: 0 4px 15px rgba(0,0,0,0.2);">
        <div style="font-size: 3em; margin-bottom: 10px;">📊</div>
        <div style="font-size: 2.5em; font-weight: bold; margin: 10px 0;">{count:,}</div>
        <div style="font-size: 1.2em; opacity: 0.9;">{label}</div>
    </div>
    """

def create_column_info(df):
    """Create a summary of DataFrame structure."""
    info_html = "<div class='column-info'><h4>📋 Dataset Structure</h4><ul>"
    for col in df.columns:
        dtype = str(df[col].dtype)
        non_null = df[col].count()
        info_html += f"<li><strong>{col}</strong> ({dtype}) - {non_null:,} non-null values</li>"
    info_html += "</ul></div>"
    
    style = """
    <style>
        .column-info {
            background: #f8f9fa; padding: 20px; border-radius: 10px; 
            border-left: 4px solid #007bff; margin: 15px 0;
        }
        .column-info h4 { color: #007bff; margin-top: 0; }
        .column-info ul { margin: 15px 0; }
        .column-info li { margin: 8px 0; padding: 5px 0; }
    </style>
    """
    return style + info_html

def create_text_response(text):
    """Format simple text responses."""
    return f"""
    <div style="background: linear-gradient(135deg, #6f42c1 0%, #e83e8c 100%); 
                color: white; padding: 20px; border-radius: 12px; margin: 15px 0;">
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
                    st.session_state.messages.append({"role": "user", "content": query})
                    
                    try:
                        with st.spinner("🤖 Analyzing your data..."):
                            # Get the underlying DataFrame
                            pandas_df = st.session_state.df.dataframe if hasattr(st.session_state.df, 'dataframe') else st.session_state.df
                            
                            # Try to handle the query directly first
                            handled, direct_result = smart_query_router(query, pandas_df)
                            
                            if handled:
                                # We handled it directly with clean formatting
                                st.session_state.messages.append({"role": "assistant", "content": direct_result})
                            else:
                                # Let PandasAI handle complex queries, but format the result
                                import pandasai as pai
                                result = st.session_state.df.chat(query)
                                
                                # For PandasAI results, create a simple formatted response
                                if isinstance(result, pd.DataFrame):
                                    formatted_result = create_smart_table(result, "Analysis Result")
                                elif isinstance(result, (int, float)):
                                    formatted_result = create_count_card(result, "Query Result")
                                else:
                                    # For string results, just format nicely
                                    formatted_result = create_text_response(str(result))
                                
                                st.session_state.messages.append({"role": "assistant", "content": formatted_result})
                        
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
