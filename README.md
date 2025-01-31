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
git clone https://github.com/yourusername/mushkil-viz.git
cd mushkil-viz

# Create and activate conda environment
conda env create -f environment.yml
conda activate mushkil-viz

# Alternatively, use mamba for faster installation
mamba env create -f environment.yml
mamba activate mushkil-viz
```

The environment includes all necessary dependencies for development and usage, including:
- Core analysis: pandas, numpy, scikit-learn
- Visualization: plotly, seaborn, altair
- Machine Learning: xgboost, pycaret, umap-learn
- Development tools: pytest, black, flake8

## Quick Start

1. **Basic Usage**

```bash
# Analyze a dataset with automatic domain detection
python -m mushkil_viz.cli analyze data.csv

# Specify domain and custom configuration
python -m mushkil_viz.cli analyze data.csv --domain financial --config custom_config.yaml
```

2. **Python API**

```python
from mushkil_viz.core.engine import MushkilVizEngine
from mushkil_viz.adapters.financial import FinancialAnalyzer, FinancialVisualizer

# Initialize components
analyzer = FinancialAnalyzer()
visualizer = FinancialVisualizer()

# Load and analyze data
df = pd.read_csv("financial_data.csv")
analysis_results = analyzer.analyze(df)
visualizations = visualizer.visualize(df, analysis_results)
```

## Project Structure

```
mushkil_viz/
├── core/                 # Core analysis components
├── adapters/            # Domain-specific adapters
├── configs/             # Configuration files
├── examples/            # Example datasets
├── tests/              # Test suite
└── cli.py              # Command-line interface
```

## Domain Support

### Financial Analysis
- Transaction pattern analysis
- Spending categorization
- Cash flow analysis
- Recurring transaction detection
- Merchant analysis

### Coming Soon
- Biological data analysis
- Real estate market analysis
- Time series analysis
- Text data analysis

## Configuration

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