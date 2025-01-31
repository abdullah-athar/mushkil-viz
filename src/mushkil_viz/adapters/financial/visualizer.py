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
            
        # Extract monthly spending data and create proper datetime index
        monthly_data = pd.DataFrame(spending_patterns["monthly_spending"])
        
        # Convert tuple index (year, month) to datetime
        monthly_data.index = pd.to_datetime([f"{year}-{month:02d}-01" 
                                           for year, month in monthly_data.index])
        
        # Sort by date
        monthly_data = monthly_data.sort_index()
        
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
        
        # Update x-axis to show formatted dates
        fig.update_xaxes(
            tickformat="%b %Y",
            tickangle=45,
            row=1, col=1
        )
        fig.update_xaxes(
            tickformat="%b %Y",
            tickangle=45,
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
        recurring_data = recurring_transactions["recurring_transactions"]
        
        # Convert dictionary to DataFrame
        data_list = []
        for key, value in recurring_data.items():
            # Split the combined key into merchant and category
            identifiers = key.split('_')
            merchant = identifiers[0] if len(identifiers) > 0 else ''
            category = identifiers[1] if len(identifiers) > 1 else ''
            
            row = {
                'merchant': merchant,
                'category': category,
                'frequency': value['frequency'],
                'avg_amount': value['avg_amount'],
                'std_amount': value['std_amount']
            }
            data_list.append(row)
        
        df = pd.DataFrame(data_list)
        
        # Create hover text based on available information
        hover_texts = []
        for _, row in df.iterrows():
            hover_parts = []
            if row['merchant']:
                hover_parts.append(f"Merchant: {row['merchant']}")
            if row['category']:
                hover_parts.append(f"Category: {row['category']}")
            hover_parts.extend([
                f"Frequency: {row['frequency']}",
                f"Avg Amount: ${row['avg_amount']:.2f}",
                f"Std Dev: ${row['std_amount']:.2f}"
            ])
            hover_texts.append("<br>".join(hover_parts))
        
        # Create bubble chart
        fig = go.Figure(go.Scatter(
            x=df['avg_amount'],
            y=df['frequency'],
            mode='markers',
            marker=dict(
                size=df['frequency'] * 3,  # Size based on frequency
                sizemode='area',
                sizeref=2.*max(df['frequency'])/(40.**2),
                color=df['std_amount'],  # Color based on variability
                colorscale='Viridis',
                showscale=True,
                colorbar=dict(title='Amount Variability')
            ),
            text=hover_texts,
            hoverinfo='text'
        ))
        
        fig.update_layout(
            title="Recurring Transaction Patterns",
            xaxis_title="Average Transaction Amount ($)",
            yaxis_title="Frequency (Number of Transactions)",
            template="plotly_white",
            height=600,
            showlegend=False
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