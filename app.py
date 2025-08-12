import streamlit as st
import pandas as pd
import numpy as np
import shutil
from pathlib import Path
import pandasai
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
if 'smart_df' not in st.session_state:
    st.session_state.smart_df = None
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
        smart_df = pai.read_csv(redash_api_url)
        
        return smart_df
    except Exception as e:
        st.error(f"Error loading data from Redash: {str(e)}")
        return None


def display_data_summary(smart_df):
    """Display comprehensive data summary and statistics using PandasAI"""
    if smart_df is None:
        return
    
    st.markdown("## 📊 Data Summary & Statistics")
    
    # Get the underlying pandas DataFrame for analysis
    pandas_smart_df = smart_df.dataframe if hasattr(smart_df, 'dataframe') else smart_df
    
    # Basic metrics in cards
    col1, col2, col3, col4 = st.columns(4)
    
    with col1:
        st.markdown(f"""
        <div class="metric-card">
            <div class="metric-label">Total Records</div>
            <div class="metric-value">{len(pandas_smart_df):,}</div>
        </div>
        """, unsafe_allow_html=True)
    
    with col2:
        st.markdown(f"""
        <div class="metric-card">
            <div class="metric-label">Total Columns</div>
            <div class="metric-value">{len(pandas_smart_df.columns)}</div>
        </div>
        """, unsafe_allow_html=True)
    
    with col3:
        numeric_cols = pandas_smart_df.select_dtypes(include=[np.number]).columns
        st.markdown(f"""
        <div class="metric-card">
            <div class="metric-label">Numeric Columns</div>
            <div class="metric-value">{len(numeric_cols)}</div>
        </div>
        """, unsafe_allow_html=True)
    
    with col4:
        missing_percentage = (pandas_smart_df.isnull().sum().sum() / (len(pandas_smart_df) * len(pandas_smart_df.columns))) * 100
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
        st.dataframe(pandas_smart_df.head(20), use_container_width=True)
        
        st.markdown("### Data Shape")
        st.info(f"Dataset contains **{len(pandas_smart_df)} rows** and **{len(pandas_smart_df.columns)} columns**")
    
    with tab2:
        st.markdown("### Statistical Summary")
        if len(numeric_cols) > 0:
            st.dataframe(pandas_smart_df[numeric_cols].describe(), use_container_width=True)
        else:
            st.info("No numeric columns found for statistical analysis")
        
        st.markdown("### Missing Values Analysis")
        missing_data = pandas_smart_df.isnull().sum()
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
        dtype_counts = pandas_smart_df.dtypes.value_counts()
        
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
            dtype_smart_df = pd.DataFrame({
                'Column': pandas_smart_df.columns,
                'Data Type': pandas_smart_df.dtypes.astype(str),
                'Non-Null Count': pandas_smart_df.count(),
                'Null Count': pandas_smart_df.isnull().sum()
            })
            st.dataframe(dtype_smart_df, use_container_width=True)
    
    with tab4:
        st.markdown("### Data Visualizations")
        
        if len(numeric_cols) > 0:
            # Correlation heatmap
            if len(numeric_cols) > 1:
                st.markdown("#### Correlation Matrix")
                corr_matrix = pandas_smart_df[numeric_cols].corr()
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
                            pandas_smart_df,
                            x=selected_col,
                            title=f"Distribution of {selected_col}",
                            nbins=30
                        )
                        st.plotly_chart(fig, use_container_width=True)
                    
                    with col2:
                        fig = px.box(
                            pandas_smart_df,
                            y=selected_col,
                            title=f"Box Plot of {selected_col}"
                        )
                        st.plotly_chart(fig, use_container_width=True)

def reformat_output_with_llm(raw_response, user_query, openai_api_key):
      # Enhanced prompt for table detection and creation
        prompt = ChatPromptTemplate.from_messages(
    [
        SystemMessage(content=(
            "You are a text formatter. Your job is:\n"
            "1. Detect if the provided raw text \"{raw_response}\" contains repeated sequences of values (e.g., word(s) + number).\n"
            "2. If yes, output ONLY a Markdown table with one row per sequence, no explanations.\n"
            "3. If no structured pattern is found, output the text exactly as given.\n\n"
            "Rules:\n"
            "- Do NOT add column headers unless they are clearly present in the text.\n"
            "- Keep the structure general — do not assume specific column names.\n"
            "- No extra commentary, no rewording, no text outside the table."
        )),
        HumanMessage(content=f"User Query: {user_query}\nRaw Output: {raw_response}\n\n---\nReturn ONLY a Table (Markdown) or concise answer. If table is too wide, include only meaningful columns.")
    ]
)

    
        llm = ChatOpenAI(
            model="gpt-4o",  # Fixed: gpt-4.5 doesn't exist, using gpt-4o instead
            api_key=openai_api_key,
            temperature=0.3,
            max_tokens=1000  # Added max_tokens to prevent overly long responses
        )
    
        try:
            response = llm.invoke(prompt)
            return response.content
        except Exception as e:
            print(f"LLM reformatting failed: {e}")
            return raw_response  # Fallback to original response

def detect_and_convert_to_table(content):
    """
    Detect patterns in text that should be converted to markdown tables
    
    Handles both multi-line and single-line data formats:
    - Multi-line: "Header1 Header2\nValue1 Value2\nValue3 Value4"
    - Single-line: "Header1 Header2 Header3 Value1 Value2 Value3 Value4 Value5 Value6"
    """
    
    if not isinstance(content, str):
        return content, False
    
    content = content.strip()
    
    # First, try multi-line format (original logic)
    lines = [line.strip() for line in content.split('\n') if line.strip()]
    
    if len(lines) >= 3:  # Multi-line format
        first_line_words = lines[0].split()
        
        if len(first_line_words) >= 2:
            num_columns = len(first_line_words)
            data_rows = []
            
            # Try to parse subsequent lines
            for i in range(1, len(lines)):
                tokens = lines[i].split()
                if len(tokens) >= num_columns:
                    if num_columns == 2:
                        try:
                            float(tokens[-1])
                            city_parts = tokens[:-1]
                            count = tokens[-1]
                            row_data = [' '.join(city_parts), count]
                            data_rows.append(row_data)
                            continue
                        except ValueError:
                            pass
                    
                    if len(tokens) == num_columns:
                        data_rows.append(tokens)
                        continue
                    
                    if len(tokens) > num_columns:
                        num_data_cols = num_columns - 1
                        first_col = ' '.join(tokens[:-num_data_cols])
                        data_cols = tokens[-num_data_cols:]
                        row_data = [first_col] + data_cols
                        data_rows.append(row_data)
                        continue
                
                return content, False
            
            if len(data_rows) >= 2:
                return create_markdown_table(first_line_words, data_rows), True
    
    # If multi-line didn't work, try single-line format
    tokens = content.split()
    
    if len(tokens) < 6:  # Need at least headers + 2 data rows worth of data
        return content, False
    
    # Try different column counts (2, 3, 4, 5, etc.)
    for num_cols in range(2, 8):  # Try 2 to 7 columns
        if len(tokens) < num_cols + (num_cols * 2):  # Need headers + at least 2 data rows
            continue
            
        headers = tokens[:num_cols]
        remaining_tokens = tokens[num_cols:]
        
        # Check if remaining tokens can be evenly divided into rows
        if len(remaining_tokens) % num_cols != 0:
            continue
        
        num_data_rows = len(remaining_tokens) // num_cols
        
        if num_data_rows < 2:  # Need at least 2 data rows
            continue
        
        # Try to parse the data
        data_rows = []
        success = True
        
        for row_idx in range(num_data_rows):
            start_idx = row_idx * num_cols
            end_idx = start_idx + num_cols
            row_tokens = remaining_tokens[start_idx:end_idx]
            
            # Special handling for common patterns
            if num_cols == 3:
                # Pattern like: "City YearMonth Count"
                # Check if last token is numeric and middle could be date-like
                try:
                    float(row_tokens[-1])  # Last should be numeric
                    # Middle token should look like date (contains hyphen or numbers)
                    if '-' in row_tokens[1] or row_tokens[1].isdigit():
                        data_rows.append(row_tokens)
                        continue
                except ValueError:
                    pass
            
            elif num_cols == 2:
                # Pattern like: "City Count"
                try:
                    float(row_tokens[-1])  # Last should be numeric
                    data_rows.append(row_tokens)
                    continue
                except ValueError:
                    pass
            
            # General case: just add the tokens as they are
            data_rows.append(row_tokens)
        
        # If we successfully parsed all rows, create the table
        if len(data_rows) == num_data_rows and len(data_rows) >= 2:
            return create_markdown_table(headers, data_rows), True
    
    # Special handling for your specific case: "Destination City YearMonth Total_Deliveries"
    # This is 3 columns but first column has space in it
    if len(tokens) >= 8:  # At least 4 headers + 4 data values
        # Try pattern: "Word1 Word2 Word3 Word4" as headers, then groups of 4
        potential_headers = tokens[:4]
        remaining = tokens[4:]
        
        # Check if we can group remaining tokens into sets of 4
        if len(remaining) % 4 == 0 and len(remaining) >= 8:
            num_rows = len(remaining) // 4
            data_rows = []
            
            for i in range(num_rows):
                start_idx = i * 4
                row_data = remaining[start_idx:start_idx + 4]
                
                # For this specific pattern, combine first two as city name
                city_name = row_data[0] + ' ' + row_data[1] if row_data[0] != row_data[1] else row_data[0]
                year_month = row_data[2] if len(row_data) > 2 else row_data[1]
                deliveries = row_data[3] if len(row_data) > 3 else row_data[2]
                
                # But wait, let's check the actual pattern in your data
                # "Destination City YearMonth Total_Deliveries Faisalabad 2025-03 1450"
                # Headers: ["Destination", "City", "YearMonth", "Total_Deliveries"] - but this should be 3 cols
                # Let's try: ["Destination City", "YearMonth", "Total_Deliveries"] - 3 columns
                pass
    
    # Try 3-column format with first column having space
    if "Destination City" in content and "YearMonth" in content and "Total_Deliveries" in content:
        # Find where headers end and data begins
        header_part = "Destination City YearMonth Total_Deliveries"
        if content.startswith(header_part):
            data_part = content[len(header_part):].strip()
            data_tokens = data_part.split()
            
            # Group into sets of 3: [City, YearMonth, Count]
            if len(data_tokens) % 3 == 0 and len(data_tokens) >= 6:
                headers = ["Destination City", "YearMonth", "Total_Deliveries"]
                data_rows = []
                
                for i in range(0, len(data_tokens), 3):
                    if i + 2 < len(data_tokens):
                        city = data_tokens[i]
                        year_month = data_tokens[i + 1]
                        deliveries = data_tokens[i + 2]
                        data_rows.append([city, year_month, deliveries])
                
                if len(data_rows) >= 2:
                    return create_markdown_table(headers, data_rows), True
    
    return content, False

def create_markdown_table(headers, data_rows):
    """Helper function to create markdown table from headers and data rows"""
    # Create header row
    header_row = '| ' + ' | '.join(headers) + ' |'
    separator_row = '|' + '|'.join(['---' for _ in headers]) + '|'
    
    # Create data rows
    table_rows = []
    for row_data in data_rows:
        # Ensure we have the right number of columns
        while len(row_data) < len(headers):
            row_data.append('')
        row_data = row_data[:len(headers)]  # Trim if too many
        
        table_row = '| ' + ' | '.join(str(cell) for cell in row_data) + ' |'
        table_rows.append(table_row)
    
    # Combine all parts
    markdown_table = '\n'.join([header_row, separator_row] + table_rows)
    return markdown_table

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
                        smart_df = pai.read_csv(uploaded_file)
                        st.session_state.smart_df = smart_df
                        st.success(f"✅ File uploaded! {len(smart_df)} rows loaded.")
                    except Exception as e:
                        st.error(f"Error loading file: {str(e)}")
            
            elif data_source == "Redash API URL":
                redash_url = st.text_input(
                    "Redash API URL:",
                    placeholder="Enter Redash API URL..."
                )
                
                if redash_url and st.button("📥 Load Data"):
                    with st.spinner("Loading data from Redash..."):
                        smart_df = load_data_from_redash(redash_url)
                        if smart_df is not None:
                            st.session_state.smart_df = smart_df
                            st.success(f"✅ Data loaded! {len(smart_df)} rows loaded.")
        
        # Display current status
        st.markdown("---")
        st.markdown("## 📊 Status")
        
        agent_status = "✅ Ready" if st.session_state.agent_initialized else "❌ Not initialized"
        data_status = "✅ Loaded" if st.session_state.smart_df is not None else "❌ No data"
        
        st.markdown(f"**Agent:** {agent_status}")
        st.markdown(f"**Data:** {data_status}")
        
        if st.session_state.smart_df is not None:
            st.markdown(f"**Rows:** {len(st.session_state.smart_df):,}")
            st.markdown(f"**Columns:** {len(st.session_state.smart_df.columns)}")

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
        if st.session_state.smart_df is not None:
            tab1, tab2 = st.tabs(["💬 Chat Interface", "📊 Data Summary"])
            
            with tab1:
                st.markdown("## 💬 Chat with Your Data")
                
                # Display chat messages
                for message in st.session_state.messages:
                    with st.chat_message(message["role"]):
                        content = message["content"]
                        message_type = message.get("type", "text")
                        
                        # Check if it's already marked as a table or is already a markdown table
                        is_existing_table = (message_type == "table")
                        
                        # Try to detect and convert patterns to tables
                        converted_content, is_converted_table = detect_and_convert_to_table(content)
                        
                        # Determine if we should display as table
                        should_display_as_table = is_existing_table or is_converted_table
                        
                        if should_display_as_table:
                            # Apply custom CSS for table styling
                            st.markdown("""
                            <style>
                            .stMarkdown table {
                                width: 100%;
                                border-collapse: collapse;
                                margin: 1rem 0;
                                font-size: 14px;
                            }
                            .stMarkdown th {
                                background-color: #f0f2f6;
                                color: #262730;
                                font-weight: 600;
                                padding: 0.5rem 1rem;
                                text-align: left;
                                border-bottom: 2px solid #e6e9ef;
                            }
                            .stMarkdown td {
                                padding: 0.5rem 1rem;
                                border-bottom: 1px solid #e6e9ef;
                                color: #262730;
                            }
                            .stMarkdown tr:nth-child(even) {
                                background-color: #f9f9fb;
                            }
                            .stMarkdown tr:hover {
                                background-color: #f0f2f6;
                            }
                            </style>
                            """, unsafe_allow_html=True)
                            
                            # Display the table (use converted content if available)
                            table_content = converted_content if is_converted_table else content
                            st.markdown(table_content)
                            
                            # Show table info
                            if table_content.count('\n') > 1:
                                row_count = table_content.count('\n') - 1  # Subtract header row
                                st.caption(f"📊 Showing {row_count} rows")
                                
                            # Optional: Show original text if converted
                            if is_converted_table:
                                with st.expander("View Original Response"):
                                    st.text(content)
                                    
                        else:
                            # Display as regular text
                            st.write(content)
                
                # Chat input
                if query := st.chat_input("Ask me anything about your data..."):
                    st.session_state.messages.append({"role": "user", "content": query})
                    try:
                        with st.spinner("🤖 Analyzing your data..."):
                            import pandasai as pai
                            import pandas as pd
                            
                            result = st.session_state.smart_df.chat(query)
                            
                            if isinstance(result, pd.DataFrame):
                                # Direct DataFrame response
                                markdown_table = result.head(20).to_markdown(index=False)
                                st.session_state.messages.append({
                                    "role": "assistant",
                                    "content": markdown_table,
                                    "type": "table"
                                })
                                
                            elif isinstance(result, str):
                                # First, try pattern detection on raw result
                                converted_content, is_pattern_table = detect_and_convert_to_table(result)
                                
                                if is_pattern_table:
                                    # Pattern detected, save as table
                                    st.session_state.messages.append({
                                        "role": "assistant",
                                        "content": converted_content,
                                        "type": "table"
                                    })
                                elif '|' in result and result.count('\n') > 1:
                                    # Already a markdown table
                                    st.session_state.messages.append({
                                        "role": "assistant",
                                        "content": result,
                                        "type": "table"
                                    })
                                elif len(result) > 30:
                                    # Use LLM reformatting for longer responses
                                    formatted = reformat_output_with_llm(
                                        raw_response=result,
                                        user_query=query,
                                        openai_api_key=st.session_state.api_key
                                    )
                                    
                                    # Check if LLM created a table
                                    if '|' in formatted and formatted.count('\n') > 1 and formatted.count('|') > 2:
                                        st.session_state.messages.append({
                                            "role": "assistant",
                                            "content": formatted,
                                            "type": "table"
                                        })
                                    else:
                                        # Try pattern detection on LLM formatted response
                                        converted_formatted, is_formatted_pattern = detect_and_convert_to_table(formatted)
                                        if is_formatted_pattern:
                                            st.session_state.messages.append({
                                                "role": "assistant",
                                                "content": converted_formatted,
                                                "type": "table"
                                            })
                                        else:
                                            st.session_state.messages.append({
                                                "role": "assistant",
                                                "content": formatted,
                                                "type": "text"
                                            })
                                else:
                                    # Short string responses
                                    st.session_state.messages.append({
                                        "role": "assistant",
                                        "content": result,
                                        "type": "text"
                                    })
                            else:
                                # Handle other data types
                                st.session_state.messages.append({
                                    "role": "assistant",
                                    "content": str(result),
                                    "type": "text"
                                })
                            
                            st.rerun()
                            
                    except Exception as e:
                        error_msg = f"Sorry, I encountered an error: {str(e)}"
                        st.session_state.messages.append({"role": "assistant", "content": error_msg})
                        st.rerun()

            
            with tab2:
                display_data_summary(st.session_state.smart_df)
        
        else:
            st.markdown("""
            <div class="upload-section">
                <h3>📁 No Data Loaded</h3>
                <p>Please upload a CSV file or provide a Redash API URL from the sidebar to start analyzing your data.</p>
            </div>
            """, unsafe_allow_html=True)

if __name__ == "__main__":
    main()

