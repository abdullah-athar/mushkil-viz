from typing import Dict, List, Optional
import pandas as pd
import numpy as np
from ...core.analyzers.base import BaseAnalyzer

class FinancialAnalyzer(BaseAnalyzer):
    """Financial domain-specific analyzer."""
    
    def __init__(self):
        super().__init__()
        self.transaction_columns = []
        self.merchant_columns = []
        self.category_columns = []
        self.date_columns = []
        
    def _detect_financial_columns(self, df: pd.DataFrame) -> None:
        """Detect financial-specific columns based on name patterns."""
        for col in df.columns:
            col_lower = col.lower()
            
            # Transaction amount columns
            if any(term in col_lower for term in ["amount", "transaction", "payment", "price"]):
                self.transaction_columns.append(col)
                
            # Merchant columns    
            elif any(term in col_lower for term in ["merchant", "vendor", "payee", "recipient"]):
                self.merchant_columns.append(col)
                
            # Category columns
            elif any(term in col_lower for term in ["category", "type", "description"]):
                self.category_columns.append(col)
                
            # Date columns
            elif any(term in col_lower for term in ["date", "time", "timestamp"]):
                self.date_columns.append(col)
                
    def _compute_spending_patterns(self, df: pd.DataFrame) -> Dict:
        """Analyze spending patterns across different dimensions."""
        patterns = {}
        
        # Ensure we have transaction amounts
        if not self.transaction_columns:
            return patterns
            
        main_amount_col = self.transaction_columns[0]
        
        # Temporal analysis if date column exists
        if self.date_columns:
            date_col = self.date_columns[0]
            df[date_col] = pd.to_datetime(df[date_col])
            
            # Daily spending
            daily_spending = df.groupby(df[date_col].dt.date)[main_amount_col].agg([
                "count", "sum", "mean", "std"
            ]).to_dict()
            patterns["daily_spending"] = daily_spending
            
            # Monthly spending
            monthly_spending = df.groupby([
                df[date_col].dt.year,
                df[date_col].dt.month
            ])[main_amount_col].agg([
                "count", "sum", "mean", "std"
            ]).to_dict()
            patterns["monthly_spending"] = monthly_spending
            
        # Category analysis if category column exists
        if self.category_columns:
            cat_col = self.category_columns[0]
            category_spending = df.groupby(cat_col)[main_amount_col].agg([
                "count", "sum", "mean", "std"
            ]).to_dict()
            patterns["category_spending"] = category_spending
            
        # Merchant analysis if merchant column exists
        if self.merchant_columns:
            merch_col = self.merchant_columns[0]
            merchant_spending = df.groupby(merch_col)[main_amount_col].agg([
                "count", "sum", "mean", "std"
            ]).sort_values("sum", ascending=False).head(10).to_dict()
            patterns["top_merchants"] = merchant_spending
            
        return patterns
        
    def _detect_recurring_transactions(
        self,
        df: pd.DataFrame,
        amount_threshold: float = 0.1,
        frequency_threshold: int = 2
    ) -> Dict:
        """Detect potentially recurring transactions based on amount and frequency."""
        if not (self.transaction_columns and self.date_columns):
            return {}
            
        amount_col = self.transaction_columns[0]
        date_col = self.date_columns[0]
        
        # Convert to datetime if needed
        df[date_col] = pd.to_datetime(df[date_col])
        
        # Group similar amounts
        amount_groups = df.groupby(
            pd.cut(df[amount_col], bins=100)
        ).agg({
            date_col: "count",
            amount_col: "mean"
        })
        
        print(amount_groups)
        # Filter potential recurring transactions
        recurring = amount_groups[
            amount_groups[date_col] >= frequency_threshold
        ].to_dict()

        print(recurring.keys())
        
        return {"recurring_transactions": recurring}
        
    def _analyze_cash_flow(self, df: pd.DataFrame) -> Dict:
        """Analyze cash flow patterns and trends."""
        if not self.transaction_columns:
            return {}
            
        amount_col = self.transaction_columns[0]
        
        # Split into inflows and outflows
        inflows = df[df[amount_col] > 0][amount_col]
        outflows = df[df[amount_col] < 0][amount_col]
        
        cash_flow = {
            "total_inflow": float(inflows.sum()),
            "total_outflow": float(outflows.sum()),
            "net_flow": float(inflows.sum() + outflows.sum()),
            "inflow_stats": {
                "count": len(inflows),
                "mean": float(inflows.mean()),
                "std": float(inflows.std()),
                "median": float(inflows.median())
            },
            "outflow_stats": {
                "count": len(outflows),
                "mean": float(outflows.mean()),
                "std": float(outflows.std()),
                "median": float(outflows.median())
            }
        }
        
        return cash_flow
        
    def analyze(self, df: pd.DataFrame) -> Dict:
        """Perform comprehensive financial analysis of the dataset."""
        # Run base analysis first
        analysis_results = super().analyze(df)
        
        # Detect financial columns
        self._detect_financial_columns(df)
        
        
        # Add financial-specific analyses
        analysis_results.update({
            "financial_columns": {
                "transaction": self.transaction_columns,
                "merchant": self.merchant_columns,
                "category": self.category_columns,
                "date": self.date_columns
            },
            "spending_patterns": self._compute_spending_patterns(df),
            "recurring_transactions": self._detect_recurring_transactions(df),
            "cash_flow_analysis": self._analyze_cash_flow(df)
        })
        
        return analysis_results 