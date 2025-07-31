# 🚛 Truck It Inlytics - AI-Powered Data Analytics Chatbot

An AI-powered data analytics chatbot built with Streamlit and PandasAI. Transform your data analysis workflow with natural language queries, stunning visualizations, and intelligent insights—all wrapped in a beautiful, user-friendly interface.

### ✨ What Makes It Special?

- 🤖 **Conversational AI**: Chat with your data using natural language queries
- 🎨 **Beautiful UI**: Gradient-based design with professional aesthetics
- 🔄 **Multi-LLM Support**: Choose between OpenAI and Groq for AI processing
- 📊 **Interactive Visualizations**: Comprehensive data insights with Plotly charts
- 📁 **Flexible Data Sources**: Upload CSV files or connect to Redash APIs
- 🚀 **Real-time Analysis**: Instant responses to your data questions

## 🎯 Key Features

### 🤖 Intelligent Chat Interface
- Natural language data queries powered by PandasAI
- Human and AI avatars for clear conversation flow
- Real-time response generation with loading indicators
- Error handling with user-friendly messages

### 📊 Comprehensive Data Analytics
- **Statistical Summaries**: Detailed descriptive statistics for numeric columns
- **Missing Value Analysis**: Visual identification of data quality issues
- **Data Type Profiling**: Automatic categorization and analysis of column types
- **Correlation Analysis**: Interactive heatmaps for numeric relationships
- **Distribution Plots**: Histograms and box plots for data exploration

### 🎨 Professional UI/UX
- Gradient color schemes with modern design principles
- Card-based metrics display for quick insights
- Tabbed interface for organized data exploration
- Responsive design that works on all screen sizes
- Custom CSS styling for enhanced aesthetics

### 🔧 Technical Excellence
- **Multi-LLM Architecture**: Seamless switching between OpenAI and Groq
- **Secure API Key Management**: Password-protected key input
- **Error Resilience**: Comprehensive exception handling
- **Performance Optimized**: Efficient data processing and visualization

## 🚀 Quick Start

### Prerequisites
- Python 3.8+
- OpenAI API Key or Groq API Key

### Installation

1. **Clone the repository**
```bash
git clone https://github.com/yourusername/truck-it-inlytics.git
cd truck-it-inlytics
```

2. **Install dependencies**
```bash
pip install -r requirements.txt
```

3. **Run the application**
```bash
streamlit run app.py
```

4. **Open your browser** and navigate to `http://localhost:8501`

## 📋 Requirements

```txt
streamlit
pandasai
pandasai-openai
langchain-groq
numpy
plotly
requests
python-dotenv
```

## 🎮 How to Use

### Step 1: Configuration
1. Select your preferred LLM provider (OpenAI or Groq)
2. Enter your API key securely
3. Click "Initialize Agent" to set up the AI

### Step 2: Data Loading
Choose your data source:
- **CSV Upload**: Drag and drop your CSV file
- **Redash API**: Provide your Redash API URL for live data

### Step 3: Analysis
- **Chat Interface**: Ask questions about your data in natural language
- **Data Summary**: Explore comprehensive statistics and visualizations

## 💡 Example Queries

```
"What are the top 5 cities by delivery volume?"
"Show me the correlation between delivery time and customer satisfaction"
"Which riders have the highest completion rates?"
"What's the trend in deliveries over the past month?"
"Identify any patterns in delivery delays"
```

## 🏗️ Architecture

```
┌─────────────────┐    ┌──────────────────┐    ┌─────────────────┐
│   Streamlit UI  │    │   PandasAI Core  │    │  LLM Providers  │
│                 │◄──►│                  │◄──►│                 │
│  • Chat Interface│    │  • Data Processing│   │  • OpenAI       │
│  • Visualizations│   │  • Query Engine  │    │  • Groq         │
│  • File Upload  │    │  • Response Gen. │    │                 │
└─────────────────┘    └──────────────────┘    └─────────────────┘
         │                        │                        │
         ▼                        ▼                        ▼
┌─────────────────┐    ┌──────────────────┐    ┌─────────────────┐
│  Data Sources   │    │   Plotly Charts  │    │   Session State │
│                 │    │                  │    │                 │
│  • CSV Files    │    │  • Correlations  │    │  • Chat History │
│  • Redash APIs  │    │  • Distributions │    │  • Data Cache   │
│  • Live Data    │    │  • Statistics    │    │  • Config       │
└─────────────────┘    └──────────────────┘    └─────────────────┘
```

## 🎨 Screenshots

### Dashboard Overview
![Dashboard](https://via.placeholder.com/800x400/667eea/ffffff?text=Beautiful800x400/f093fb/ffffff?text=Conversationalizations
![Charts](https://via.placeholder.com/800x400/a8edea/333333?text=Interactive

### Environment Variables
```bash
OPENAI_API_KEY=your_openai_key_here
GROQ_API_KEY=your_groq_key_here
```

### Customization
- Modify color schemes in the CSS section
- Add new chart types in `display_data_summary()`
- Extend LLM support by adding new providers
- Customize chat avatars and styling

## 🤝 Contributing

We welcome contributions! Here's how you can help:

1. **Fork the repository**
2. **Create a feature branch** (`git checkout -b feature/amazing-feature`)
3. **Commit your changes** (`git commit -m 'Add amazing feature'`)
4. **Push to the branch** (`git push origin feature/amazing-feature`)
5. **Open a Pull Request**

### Development Setup
```bash
# Install development dependencies
pip install -r requirements-dev.txt

# Run tests
pytest tests/

# Format code
black app.py

# Lint code
flake8 app.py
```

## 📈 Roadmap

- [ ] **Database Integration**: Support for SQL databases
- [ ] **Advanced Analytics**: Machine learning model integration
- [ ] **Export Features**: PDF/Excel report generation
- [ ] **Multi-language Support**: Internationalization
- [ ] **Real-time Streaming**: Live data dashboard updates
- [ ] **User Authentication**: Multi-user support with permissions
- [ ] **Custom Visualizations**: User-defined chart types

## 🐛 Known Issues

- Large datasets (>100MB) may cause performance issues
- Some complex SQL-like queries might need refinement
- PDF export feature is planned for future releases

## 📄 License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.

## 🙏 Acknowledgments

- **PandasAI Team** - For the amazing conversational AI framework
- **Streamlit** - For the incredible web app framework
- **Plotly** - For beautiful, interactive visualizations
- **OpenAI & Groq** - For powerful language models

## 📞 Support

Having issues? We're here to help!

- 🐛 **Bug Reports**: [Open an issue](https://github.com/yourusername/truck-it-inlytics/issues)
- 💡 **Feature Requests**: [Start a discussion](https://github.com/yourusername/truck-it-inlytics/discussions)
- 📧 **Email**: support@truckit-inlytics.com
- 💬 **Discord**: [Join our community](https://discord.gg/truckit-inlytics)


**Made with ❤️ by the Truck It Inlytics Team**
