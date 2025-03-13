from typing import Optional, List, Literal, Union
from pydantic import BaseModel, Field, ConfigDict

class VisualizationParams(BaseModel):
    """Parameters for visualization generation"""
    model_config = ConfigDict(extra='allow')
    visualization_type: Literal["histogram", "bar", "scatter", "line", "pie"] = Field(
        default="histogram", description="Type of visualization to generate")
    columns: List[str] = Field(
        default_factory=list, description="Columns to use in the visualization")
    title: Optional[str] = Field(
        default=None, description="Title for the visualization")
    
class CSVQueryResponse(BaseModel):
    """Structure for LLM responses to CSV data questions"""
    model_config = ConfigDict(extra='allow')
    answer: Union[str, dict] = Field(
        ..., description="Detailed answer to the user's question about the CSV data")
    create_visualization: bool = Field(
        default=False, description="Whether a visualization should be created")
    visualization_params: Optional[Union[VisualizationParams, dict]] = Field(
        default=None, description="Parameters for visualization if needed") 