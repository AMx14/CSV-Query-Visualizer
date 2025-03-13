import io
import pandas as pd
from pydantic_ai import Agent
from pydantic_ai.models.openai import OpenAIModel
from pydantic_ai.providers.openai import OpenAIProvider
from models import CSVQueryResponse, VisualizationParams
import json

class LLMProcessor:
    """Module for LLM integration using Ollama"""
    
    def __init__(self, model_name: str = "llama3.2:latest"):
        """Initialize the LLM processor with the specified model"""
        self.model = OpenAIModel(
            model_name=model_name,
            provider=OpenAIProvider(
                base_url='http://localhost:11434/v1',
                api_key='dummy'  # Required by the provider but not used by Ollama
            )
        )
        
        self.agent = Agent(
            self.model,
            result_type=CSVQueryResponse,
            retries=5
        )
    
    def process_query(self, df: pd.DataFrame, query: str, include_visualization: bool) -> CSVQueryResponse:
        """Process a query about the CSV data using the LLM"""
        try:
            # Prepare data description for the LLM
            buffer = io.StringIO()
            df.info(buf=buffer)
            df_info = buffer.getvalue()
            
            # Create summary statistics and sample data
            df_summary = df.describe(include='all').to_string()
            df_sample = df.head(5).to_string()
            column_names = df.columns.tolist()
            
            # Prepare categorical value distributions
            categorical_info = ""
            for col in df.columns:
                if not pd.api.types.is_numeric_dtype(df[col].dtype) or df[col].nunique() < 10:
                    value_counts = df[col].value_counts()
                    categorical_info += f"\nDistribution of {col}:\n"
                    for value, count in value_counts.items():
                        percentage = (count / len(df)) * 100
                        categorical_info += f"  {value}: {count} ({percentage:.1f}%)\n"
            
            # Construct a simpler prompt for the LLM
            prompt = f"""
            Analyze this CSV data and answer the question: "{query}"
            
            Data Overview:
            - Columns: {', '.join(column_names)}
            - Total rows: {len(df)}
            
            Summary Statistics:
            {df_summary}
            
            Sample Data:
            {df_sample}
            
            {categorical_info}
            
            Instructions:
            1. Answer the question using ONLY the data provided above
            2. Be precise with numbers and statistics
            3. If the information isn't in the data, say "I cannot answer this question with the available data"
            4. Format your response as a valid JSON object
            
            {"The user wants a visualization with the answer." if include_visualization else ""}
            
            Available visualization types:
            - "histogram": Shows distribution of a single numeric column
            - "pie": Shows distribution of a categorical column
            - "bar": Shows bar chart (can be used for comparing categories or time series)
            - "scatter": Shows relationship between two numeric columns
            - "line": Shows trend over time or ordered data
            
            Respond with a JSON object in this exact format:
            {{
                "answer": "Your answer here",
                "create_visualization": false,
                "visualization_params": {{
                    "visualization_type": "line",  # One of: histogram, pie, bar, scatter, line
                    "columns": ["column1", "column2"],  # Required columns for the visualization
                    "title": "Optional title for the visualization",
                    "x_axis_label": "Optional x-axis label",
                    "y_axis_label": "Optional y-axis label"
                }}
            }}
            
            Only set create_visualization to true if the user specifically asked for a visualization.
            When creating a visualization:
            1. Choose appropriate visualization type based on the data and question
            2. Specify exactly which columns to use
            3. Provide clear title and axis labels
            """
            
            # Run the query through the LLM agent
            response = self.agent.run_sync(prompt)
            
            # Debug print the response
            print(f"Raw response from LLM: {response}")
            
            # Handle AgentRunResult type
            if hasattr(response, 'data'):
                return response.data
            
            # Handle different response types
            if isinstance(response, str):
                try:
                    response = json.loads(response)
                except json.JSONDecodeError:
                    # If it's a string but not JSON, create a simple response
                    response = {
                        "answer": response,
                        "create_visualization": False,
                        "visualization_params": None
                    }
            
            # Ensure we have a dictionary
            if not isinstance(response, dict):
                raise ValueError(f"Unexpected response type: {type(response)}")
            
            # Create the response object
            try:
                return CSVQueryResponse(**response)
            except Exception as e:
                print(f"Error creating CSVQueryResponse: {str(e)}")
                # Create a fallback response
                return CSVQueryResponse(
                    answer=str(response),
                    create_visualization=False,
                    visualization_params=None
                )
            
        except json.JSONDecodeError as e:
            print(f"JSON decode error: {str(e)}")
            raise RuntimeError(f"LLM returned invalid JSON: {str(e)}")
        except Exception as e:
            print(f"General error: {str(e)}")
            raise RuntimeError(f"Error processing query: {str(e)}") 