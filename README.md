# MushkilViz: Intelligent Tabular Data Analysis & Visualization System

MushkilViz is a powerful Python framework for automated analysis and visualization of structured datasets. It intelligently detects data types, extracts meaningful relationships, and generates domain-aware visualizations with minimal user configuration.

## Features

- 🔍 **Automated Data Understanding**
  - Smart column type detection
  - Data quality assessment
  - Statistical profiling
  - Pattern recognition
  - Domain classification

- 📊 **Intelligent Visualization**
  - Context-aware plot selection
  - Interactive visualizations using Plotly
  - Customizable visualization templates
  - Multi-dimensional data exploration

- 🎯 **Domain-Specific Analysis**
  - Financial data analysis
  - Biological data analysis (coming soon)
  - Real estate market analysis (coming soon)
  - Extensible plugin architecture

- 📈 **Advanced Analytics**
  - Correlation analysis
  - Anomaly detection
  - Trend identification
  - Pattern mining
  - Dimensionality reduction

## Installation

```bash
# Clone the repository
git clone git@github.com:abdullah-athar/mushkil-viz.git
cd mushkil-viz

# Create and activate conda environment
conda env create -f environment.yml
conda activate mushkil

# Alternatively, use mamba for faster installation
mamba env create -f environment.yml
mamba activate mushkil

# Install deps
uv pip install -r requirements/requirements.txt 

# Copy env boilerplate and update variables
cp .env.example .env

# Set up your Google Gemini API key
echo "GOOGLE_GEMINI_KEY=your_actual_api_key_here" > .env
```

The environment includes all necessary dependencies for development and usage, including:
- Core analysis: pandas, numpy, scikit-learn
- Visualization: plotly, seaborn, altair
- Machine Learning: xgboost, pycaret, umap-learn
- Development tools: pytest, black, flake8

## Quick Start

### Web Interface (Streamlit)

```bash
# Run the Streamlit application
make run-app

# OR alternatively 
streamlit run src/mushkil_viz/streamlit/app.py --server.port=8501
```

Then open your browser to: **http://localhost:8501**

## Project Structure

```
mushkil_viz/
├── agent/
├── streamlit/
```

MushkilViz uses YAML configuration files for customization:

```yaml
# configs/domain_profiles/financial.yaml
analysis:
  outlier_detection:
    method: zscore
    threshold: 3.0
  
visualization:
  color_scheme:
    positive: "#2ecc71"
    negative: "#e74c3c"
```

## Development

1. **Setup Development Environment**

```bash
# Install additional development dependencies
conda install -c conda-forge black flake8 isort pre-commit

# Run tests
pytest tests/

# Check code style
flake8 mushkil_viz/
black mushkil_viz/
```

2. **Adding New Domains**

Create new domain adapters by extending base classes:

```python
from mushkil_viz.core.analyzers import BaseAnalyzer

class CustomDomainAnalyzer(BaseAnalyzer):
    def __init__(self):
        super().__init__()
        # Add domain-specific initialization
```

## Contributing

1. Fork the repository
2. Create a feature branch
3. Commit your changes
4. Push to the branch
5. Create a Pull Request

## License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.

## Acknowledgments

- Built with [Plotly](https://plotly.com/python/)
- Powered by [pandas](https://pandas.pydata.org/)
- Styled with [Rich](https://github.com/Textualize/rich)

## Contact

- GitHub Issues: [Report a bug](https://github.com/yourusername/mushkil-viz/issues)
- Email: ama86@cantab.ac.uk / waleedhashmi@nyu.edu
- Authors: Waleed Hashmi and Abdullah Athar