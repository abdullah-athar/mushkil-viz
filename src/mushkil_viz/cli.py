import typer
import pandas as pd
from pathlib import Path
from typing import Optional
import yaml
from rich.console import Console
from rich.progress import Progress, SpinnerColumn, TextColumn
import webbrowser
import tempfile
import json

from .core.engine import MushkilVizEngine

app = typer.Typer()
console = Console()

def load_config(config_path: Path) -> dict:
    """Load configuration from YAML file."""
    with open(config_path, 'r') as f:
        return yaml.safe_load(f)

def generate_html_report(visualizations: dict, config: dict, domain: str) -> str:
    """Generate HTML report from visualizations."""
    html_template = """
    <!DOCTYPE html>
    <html>
    <head>
        <title>MushkilViz Analysis Report</title>
        <script src="https://cdn.plot.ly/plotly-latest.min.js"></script>
        <style>
            body {
                font-family: Arial, sans-serif;
                margin: 0;
                padding: 20px;
                background-color: #f5f5f5;
            }
            .container {
                max-width: 1200px;
                margin: 0 auto;
                background-color: white;
                padding: 20px;
                border-radius: 5px;
                box-shadow: 0 2px 5px rgba(0,0,0,0.1);
            }
            .section {
                margin-bottom: 30px;
            }
            h1, h2 {
                color: #333;
            }
            .domain-info {
                background-color: #e8f5e9;
                padding: 10px;
                border-radius: 5px;
                margin-bottom: 20px;
            }
            .plot {
                margin-bottom: 20px;
                padding: 10px;
                background-color: white;
                border-radius: 5px;
            }
            .section-description {
                color: #666;
                margin-bottom: 15px;
                font-style: italic;
            }
            .plot-description {
                color: #666;
                margin: 10px 0;
                font-size: 0.9em;
            }
        </style>
    </head>
    <body>
        <div class="container">
            <h1>MushkilViz Analysis Report</h1>
            <div class="domain-info">
                <strong>Detected Domain:</strong> {{ domain }}
            </div>
            {% for section in sections %}
            <div class="section">
                <h2>{{ section.title }}</h2>
                {% if section.description %}
                <div class="section-description">{{ section.description }}</div>
                {% endif %}
                {% for plot_id in section.get('visualizations', section.get('plots', [])) %}
                    {% if visualizations.get(plot_id) %}
                    <div class="plot">
                        <div id="{{ plot_id }}"></div>
                        {% if descriptions.get(plot_id) %}
                        <div class="plot-description">{{ descriptions[plot_id] }}</div>
                        {% endif %}
                        <script>
                            var plotData = {{ visualizations[plot_id] | safe }};
                            Plotly.newPlot('{{ plot_id }}', plotData.data, plotData.layout);
                        </script>
                    </div>
                    {% endif %}
                {% endfor %}
            </div>
            {% endfor %}
        </div>
    </body>
    </html>
    """
    
    from jinja2 import Template
    template = Template(html_template)
    
    # Extract descriptions from visualization specs
    descriptions = {}
    for viz_id, viz_data in visualizations.items():
        try:
            plot_data = json.loads(viz_data)
            if isinstance(plot_data, dict) and "description" in plot_data:
                descriptions[viz_id] = plot_data["description"]
        except:
            pass
    
    return template.render(
        sections=config["report"]["sections"],
        visualizations=visualizations,
        descriptions=descriptions,
        domain=domain
    )

@app.command()
def analyze(
    input_file: Path = typer.Argument(..., help="Path to input data file"),
    domain: Optional[str] = typer.Option(None, help="Domain of the dataset (if None, will auto-detect)"),
    config_file: Optional[Path] = typer.Option(None, help="Path to configuration file"),
    output_file: Optional[Path] = typer.Option(None, help="Path to save the report")
):
    """Analyze a dataset and generate visualizations."""
    
    # Input validation
    if not input_file.exists():
        console.print(f"[red]Error: Input file {input_file} does not exist[/red]")
        raise typer.Exit(1)
        
    if config_file and not config_file.exists():
        console.print(f"[red]Error: Config file {config_file} does not exist[/red]")
        raise typer.Exit(1)
        
    with Progress(
        SpinnerColumn(),
        TextColumn("[progress.description]{task.description}"),
        console=console
    ) as progress:
        # Initialize engine
        task = progress.add_task("Initializing analysis engine...", total=None)
        engine = MushkilVizEngine(config_path=config_file)
        progress.remove_task(task)
        
        # Process dataset
        task = progress.add_task("Processing dataset...", total=None)
        #try:
        results = engine.process_dataset(input_file, domain=domain)
        # except Exception as e:
        #     progress.remove_task(task)
        #     console.print(f"[red]Error processing dataset: {str(e)}[/red]")
        #     raise typer.Exit(1)
        progress.remove_task(task)
        
        # Generate report
        task = progress.add_task("Generating report...", total=None)
        report_html = generate_html_report(
            results["visualizations"],
            load_config(config_file) if config_file else load_config(
                Path(__file__).parent / "configs" / "domain_profiles" / f"{results['domain']}.yaml"
            ),
            results['domain']
        )
        
        if output_file:
            output_path = Path(output_file)
        else:
            # Create temporary file
            temp_file = tempfile.NamedTemporaryFile(
                delete=False,
                suffix='.html',
                mode='w',
                encoding='utf-8'
            )
            output_path = Path(temp_file.name)
            
        with open(output_path, 'w', encoding='utf-8') as f:
            f.write(report_html)
            
        progress.remove_task(task)
        
    # Open report in browser
    console.print(f"\n[green]Analysis complete! Opening report in your browser...[/green]")
    webbrowser.open(f"file://{output_path.absolute()}")
    
if __name__ == "__main__":
    app() 