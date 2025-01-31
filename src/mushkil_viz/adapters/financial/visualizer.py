from typing import Dict, List, Optional, Any
import pandas as pd
import numpy as np
import plotly.graph_objects as go
import plotly.express as px
from plotly.subplots import make_subplots
from ...core.visualizers.base import BaseVisualizer

class FinancialVisualizer(BaseVisualizer):
    """Financial domain-specific visualizer."""
    
    def __init__(self):
        super().__init__()
        self.plot_registry.update({
            "spending_trends": self._plot_spending_trends,
            "category_breakdown": self._plot_category_breakdown,
            "merchant_analysis": self._plot_merchant_analysis,
            "cash_flow": self._plot_cash_flow,
            "recurring_transactions": self._plot_recurring_transactions
        })
        
    def _plot_spending_trends(
        self,
        spending_patterns: Dict[str, Any],
        **kwargs
    ) -> go.Figure:
        """Create spending trends visualization."""
        if not spending_patterns.get("monthly_spending"):
            return None
            
        # Extract monthly spending data
        monthly_data = pd.DataFrame(spending_patterns["monthly_spending"])
        
        fig = make_subplots(
            rows=2, cols=1,
            subplot_titles=("Monthly Transaction Count", "Monthly Spending Amount"),
            vertical_spacing=0.15
        )
        
        # Transaction count trend
        fig.add_trace(
            go.Scatter(
                x=monthly_data.index,
                y=monthly_data["count"],
                mode="lines+markers",
                name="Transaction Count"
            ),
            row=1, col=1
        )
        
        # Spending amount trend
        fig.add_trace(
            go.Scatter(
                x=monthly_data.index,
                y=monthly_data["sum"],
                mode="lines+markers",
                name="Total Spending"
            ),
            row=2, col=1
        )
        
        fig.update_layout(
            title="Monthly Spending Trends",
            template="plotly_white",
            height=800,
            showlegend=True
        )
        
        return fig
        
    def _plot_category_breakdown(
        self,
        spending_patterns: Dict[str, Any],
        **kwargs
    ) -> go.Figure:
        """Create category spending breakdown visualization."""
        if not spending_patterns.get("category_spending"):
            return None
            
        # Extract category spending data
        category_data = pd.DataFrame(spending_patterns["category_spending"])
        
        # Create sunburst chart
        fig = go.Figure(go.Sunburst(
            labels=category_data.index,
            parents=[""] * len(category_data),
            values=category_data["sum"],
            branchvalues="total"
        ))
        
        fig.update_layout(
            title="Spending by Category",
            template="plotly_white",
            width=800,
            height=800
        )
        
        return fig
        
    def _plot_merchant_analysis(
        self,
        spending_patterns: Dict[str, Any],
        **kwargs
    ) -> go.Figure:
        """Create merchant spending analysis visualization."""
        if not spending_patterns.get("top_merchants"):
            return None
            
        # Extract merchant data
        merchant_data = pd.DataFrame(spending_patterns["top_merchants"])
        
        # Create horizontal bar chart
        fig = go.Figure(go.Bar(
            x=merchant_data["sum"],
            y=merchant_data.index,
            orientation="h",
            text=merchant_data["sum"].round(2),
            textposition="auto"
        ))
        
        fig.update_layout(
            title="Top Merchants by Spending",
            xaxis_title="Total Spending",
            yaxis_title="Merchant",
            template="plotly_white",
            height=600
        )
        
        return fig
        
    def _plot_cash_flow(
        self,
        cash_flow_analysis: Dict[str, Any],
        **kwargs
    ) -> go.Figure:
        """Create cash flow visualization."""
        if not cash_flow_analysis:
            return None
            
        # Create waterfall chart
        fig = go.Figure(go.Waterfall(
            name="Cash Flow",
            orientation="v",
            measure=["relative", "relative", "total"],
            x=["Inflows", "Outflows", "Net Flow"],
            y=[
                cash_flow_analysis["total_inflow"],
                cash_flow_analysis["total_outflow"],
                cash_flow_analysis["net_flow"]
            ],
            connector={"line": {"color": "rgb(63, 63, 63)"}},
            decreasing={"marker": {"color": "red"}},
            increasing={"marker": {"color": "green"}},
            totals={"marker": {"color": "blue"}}
        ))
        
        fig.update_layout(
            title="Cash Flow Summary",
            template="plotly_white",
            showlegend=False,
            height=500
        )
        
        return fig
        
    def _plot_recurring_transactions(
        self,
        recurring_transactions: Dict[str, Any],
        **kwargs
    ) -> go.Figure:
        """Create recurring transactions visualization."""
        if not recurring_transactions.get("recurring_transactions"):
            return None
            
        # Extract recurring transaction data
        recurring_data = pd.DataFrame(recurring_transactions["recurring_transactions"])
        
        # Create scatter plot
        fig = go.Figure(go.Scatter(
            x=recurring_data["amount"],
            y=recurring_data["date"],
            mode="markers",
            marker=dict(
                size=recurring_data["date"] * 2,  # Size based on frequency
                color=recurring_data["amount"],
                colorscale="Viridis",
                showscale=True
            ),
            text=[f"Amount: {amt:.2f}<br>Frequency: {freq}" 
                  for amt, freq in zip(recurring_data["amount"], recurring_data["date"])],
            hoverinfo="text"
        ))
        
        fig.update_layout(
            title="Recurring Transaction Patterns",
            xaxis_title="Transaction Amount",
            yaxis_title="Frequency",
            template="plotly_white",
            height=600
        )
        
        return fig
        
    def visualize(
        self,
        data: pd.DataFrame,
        analysis_results: Dict[str, Any]
    ) -> Dict[str, Any]:
        """Generate comprehensive financial visualizations."""
        # Get base visualizations
        visualizations = super().visualize(data, analysis_results)
        
        # Add financial-specific visualizations
        financial_plots = {
            "spending_trends": self._plot_spending_trends(
                analysis_results.get("spending_patterns", {})
            ),
            "category_breakdown": self._plot_category_breakdown(
                analysis_results.get("spending_patterns", {})
            ),
            "merchant_analysis": self._plot_merchant_analysis(
                analysis_results.get("spending_patterns", {})
            ),
            "cash_flow": self._plot_cash_flow(
                analysis_results.get("cash_flow_analysis", {})
            ),
            "recurring_transactions": self._plot_recurring_transactions(
                analysis_results.get("recurring_transactions", {})
            )
        }
        
        # Convert plots to JSON and add to visualizations
        for name, plot in financial_plots.items():
            if plot is not None:
                visualizations[name] = plot.to_json()
                
        return visualizations 