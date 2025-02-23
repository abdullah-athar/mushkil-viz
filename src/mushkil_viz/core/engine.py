from typing import Dict, List, Optional, Union, Type
import pandas as pd
from pathlib import Path
import yaml

from ..core.analyzers.base import BaseAnalyzer
from ..core.visualizers.base import BaseVisualizer
from ..adapters.financial.analyzer import FinancialAnalyzer
from ..adapters.financial.visualizer import FinancialVisualizer
from ..adapters.llm.analyzer import LLMAnalyzer

class DomainDetector:
    """Detects the domain of the dataset."""
    
    def __init__(self, config_path: Optional[Path] = None):
        if config_path is None:
            config_path = Path(__file__).parent.parent / "configs" / "domain_profiles"
        self.config_path = config_path
        self.domain_rules = self._load_domain_rules()
    
    def _load_domain_rules(self) -> Dict:
        """Load domain detection rules from configuration files."""
        rules = {}
        for config_file in self.config_path.glob("*.yaml"):
            with open(config_file, 'r') as f:
                rules[config_file.stem] = yaml.safe_load(f)
        return rules
    
    def detect_domain(self, df: pd.DataFrame) -> str:
        """Detect the domain of the dataset based on column names and data patterns."""
        columns = set(col.lower() for col in df.columns)
        print(f"Analyzing columns: {columns}")  # Debug logging
        
        # Check each domain's rules
        for domain, rules in self.domain_rules.items():
            print(f"Checking rules for domain: {domain}")  # Debug logging
            
            # Check column_patterns (new format)
            if 'column_patterns' in rules:
                patterns = set()
                for pattern_group in rules['column_patterns'].values():
                    patterns.update(pattern_group)
                print(f"Checking column_patterns: {patterns}")  # Debug logging
                if any(pattern in col for col in columns for pattern in patterns):
                    return domain
                    
        # Fallback to generic if no rules match
        print("No domain rules matched, falling back to generic")  # Debug logging
        return "generic"

class MushkilVizEngine:
    """Main orchestration engine for data analysis and visualization."""
    
    def __init__(self, config_path: Optional[Path] = None):
        self.config_path = config_path or Path("configs/default_settings.yaml")
        self.domain_detector = DomainDetector()
        self.config = self._load_config()
        
        # Register available analyzers and visualizers
        self.analyzers = {
            "financial": FinancialAnalyzer,
            "llm": LLMAnalyzer,
            "generic": BaseAnalyzer  # Use base analyzer as fallback
        }
        
        self.visualizers = {
            "financial": FinancialVisualizer,
            "generic": BaseVisualizer  # Use base visualizer as fallback
        }
        
    def _load_config(self) -> Dict:
        """Load configuration settings."""
        with open(self.config_path, 'r') as f:
            return yaml.safe_load(f)
    
    def _get_analyzer(self, domain: str) -> BaseAnalyzer:
        """Factory method to get appropriate analyzer."""
        analyzer_class = self.analyzers.get(domain)
        if not analyzer_class:
            analyzer_class = self.analyzers["generic"]
        return analyzer_class()
    
    def _get_visualizer(self, domain: str) -> BaseVisualizer:
        """Factory method to get appropriate visualizer."""
        visualizer_class = self.visualizers.get(domain)
        if not visualizer_class:
            visualizer_class = self.visualizers["generic"]
        return visualizer_class()
    
    def process_dataset(
        self, 
        data: Union[pd.DataFrame, str, Path],
        domain: Optional[str] = None
    ) -> Dict:
        """
        Main entry point for processing a dataset.
        
        Args:
            data: Input dataset as DataFrame or path to file
            domain: Optional domain specification, if None will be auto-detected
            
        Returns:
            Dictionary containing analysis results and visualization specs
        """
        # Load data if path provided
        if isinstance(data, (str, Path)):
            data = pd.read_csv(data)
            
        # Detect domain if not specified
        domain = domain or self.domain_detector.detect_domain(data)
        print(f"Detected domain: {domain}")  # Debug logging
        # Get appropriate analyzers and visualizers
        analyzer = self._get_analyzer(domain)
        visualizer = self._get_visualizer(domain)
        
        # Execute analysis pipeline
        analysis_results = analyzer.analyze(data)
        visualization_results = visualizer.visualize(data, analysis_results)
        
        return {
            "domain": domain,
            "analysis_results": analysis_results,
            "visualizations": visualization_results
        }
    
    def generate_report(
        self, 
        results: Dict,
        output_format: str = "html",
        template_path: Optional[Path] = None
    ) -> Path:
        """Generate analysis report in specified format."""
        # TODO: Implement report generation using Jinja2 templates
        pass 