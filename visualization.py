import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
from models import VisualizationParams

class VisualizationGenerator:
    """Module for generating visualizations based on CSV data"""
    
    @staticmethod
    def create_visualization(df: pd.DataFrame, params: VisualizationParams) -> plt.Figure:
        """Generate a visualization based on the specified parameters"""
        fig, ax = plt.subplots(figsize=(10, 6))
        
        try:
            viz_type = params.visualization_type
            columns = params.columns
            title = params.title or f"Visualization of {', '.join(columns)}"
            
            if viz_type == "histogram" and len(columns) >= 1:
                df[columns[0]].hist(ax=ax, bins=20, edgecolor='black')
                ax.set_title(title or f"Histogram of {columns[0]}")
                ax.set_xlabel(columns[0])
                ax.set_ylabel("Frequency")
            
            elif viz_type == "pie" and len(columns) >= 1:
                value_counts = df[columns[0]].value_counts()
                wedges, texts, autotexts = ax.pie(
                    value_counts, 
                    labels=value_counts.index, 
                    autopct='%1.1f%%',
                    textprops={'fontsize': 10}
                )
                ax.set_title(title or f"Distribution of {columns[0]}")
                ax.axis('equal')
            
            elif viz_type == "bar":
                if len(columns) >= 2:
                    # Two columns: category and value
                    df.groupby(columns[0])[columns[1]].mean().plot(kind='bar', ax=ax)
                    ax.set_title(title or f"Average {columns[1]} by {columns[0]}")
                    ax.set_xlabel(columns[0])
                    ax.set_ylabel(f"Average {columns[1]}")
                elif len(columns) >= 1:
                    # One column: count by category
                    df[columns[0]].value_counts().plot(kind='bar', ax=ax)
                    ax.set_title(title or f"Count of {columns[0]}")
                    ax.set_xlabel(columns[0])
                    ax.set_ylabel("Count")
            
            elif viz_type == "scatter" and len(columns) >= 2:
                df.plot.scatter(x=columns[0], y=columns[1], ax=ax)
                ax.set_title(title or f"{columns[1]} vs {columns[0]}")
                ax.set_xlabel(columns[0])
                ax.set_ylabel(columns[1])
            
            elif viz_type == "line" and len(columns) >= 2:
                df.sort_values(by=columns[0]).plot.line(x=columns[0], y=columns[1], ax=ax, marker='o')
                ax.set_title(title or f"{columns[1]} over {columns[0]}")
                ax.set_xlabel(columns[0])
                ax.set_ylabel(columns[1])
            
            plt.tight_layout()
            return fig
        
        except Exception as e:
            # Handle visualization errors
            plt.close(fig)
            fig, ax = plt.subplots()
            ax.text(0.5, 0.5, f"Error generating visualization: {str(e)}", 
                    ha='center', va='center')
            return fig 