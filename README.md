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
  - Dark mode support with optimized color schemes
  - Responsive and adaptive layouts

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

# Set up environment configuration
cp src/mushkil_viz/frontend/.env.example src/mushkil_viz/frontend/.env
cp src/mushkil_viz/server/.env.example src/mushkil_viz/server/.env
```

The environment includes all necessary dependencies for development and usage, including:
- Core analysis: pandas, numpy, scikit-learn
- Visualization: plotly, seaborn, altair
- Machine Learning: xgboost, pycaret, umap-learn
- Development tools: pytest, black, flake8
- Web Interface: fastapi, uvicorn, react, mantine

### Environment Configuration

The project uses environment files for configuration. Example files are provided:

1. **Frontend Configuration** (`src/mushkil_viz/frontend/.env`):
```bash
# Development defaults
VITE_FRONTEND_PORT=3001
VITE_BACKEND_URL=http://localhost:8001
VITE_API_BASE_PATH=/api
```

2. **Backend Configuration** (`src/mushkil_viz/server/.env`):
```bash
# Development defaults
BACKEND_PORT=8001
ALLOWED_ORIGINS=http://localhost:3001,http://localhost:3000
DEBUG=True
```

For development:
1. Copy the `.env.example` files to `.env` in both frontend and server directories
2. Modify the values as needed for your local setup
3. The `.env` files are gitignored to prevent committing sensitive information

For production:
1. Create new `.env` files based on the examples
2. Set appropriate production values
3. Ensure proper security measures (CORS, debugging, etc.)

## Quick Start

1. **Basic Usage**

```bash
# Analyze a dataset with automatic domain detection
python -m mushkil_viz.cli analyze data.csv

# Specify domain and custom configuration
python -m mushkil_viz.cli analyze data.csv --domain financial --config custom_config.yaml
```

2. **Web Interface**

```bash
# Start the backend server
uvicorn mushkil_viz.server.main:app --reload --port 8001

# In a new terminal, start the frontend
cd src/mushkil_viz/frontend
npm install
npm run dev
```

Visit http://localhost:3001 to access the web interface:
1. Upload your CSV file using drag-and-drop or file selection
2. View interactive visualizations with domain-specific insights
3. Toggle between light and dark modes for optimal viewing experience

3. **Python API**

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
├── src/
│   └── mushkil_viz/
│       ├── core/           # Core analysis components
│       ├── adapters/       # Domain-specific adapters
│       ├── server/         # FastAPI backend server
│       ├── frontend/       # React frontend
│       ├── configs/        # Configuration files
│       └── cli.py         # Command-line interface
├── examples/              # Example datasets
└── tests/                # Test suite
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
    light:
      positive: "#2ecc71"
      negative: "#e74c3c"
    dark:
      positive: "#69DB7C"
      negative: "#FF8787"
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

3. **UI Development**

The frontend supports extensive customization:
- Light and dark mode themes with optimized visualization colors
- Responsive layouts for various screen sizes
- Custom color schemes per visualization type
- Configurable chart interactions
- Accessibility-focused design

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
- UI powered by [Mantine](https://mantine.dev/)

## Contact

- GitHub Issues: [Report a bug](https://github.com/yourusername/mushkil-viz/issues)
- Email: ama86@cantab.ac.uk / waleedhashmi@nyu.edu
- Authors: Waleed Hashmi and Abdullah Athar