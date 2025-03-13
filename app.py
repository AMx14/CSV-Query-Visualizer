import gradio as gr
from csv_handler import CSVHandler
from llm_processor import LLMProcessor
from visualization import VisualizationGenerator
from models import VisualizationParams

class CSVAnalysisApp:
    """Main application class integrating all modules"""
    
    def __init__(self):
        """Initialize the application components"""
        self.csv_handler = CSVHandler()
        self.llm_processor = LLMProcessor()
        self.viz_generator = VisualizationGenerator()
        self.current_df = None
    
    def handle_file_upload(self, file):
        """Handle CSV file upload"""
        try:
            self.current_df, info = self.csv_handler.load_csv(file.name)
            return info
        except Exception as e:
            return f"Error: {str(e)}"
    
    def handle_question(self, question, include_visualization):
        """Process a user question about the CSV data"""
        if self.current_df is None:
            return "Please upload a CSV file first.", None
        
        try:
            # Process the question using the LLM
            response = self.llm_processor.process_query(
                self.current_df, 
                question, 
                include_visualization
            )
            
            # Generate visualization if requested
            fig = None
            if response.create_visualization and response.visualization_params:
                fig = self.viz_generator.create_visualization(
                    self.current_df,
                    response.visualization_params
                )
            
            return response.answer, fig
        
        except Exception as e:
            return f"Error processing question: {str(e)}", None

# Create the Gradio interface for the application
def create_interface():
    app = CSVAnalysisApp()
    
    # Custom CSS for modern dark theme
    custom_css = """
    :root {
        --background: #0f172a;
        --foreground: #f8fafc;
        --card: #1e293b;
        --card-foreground: #f8fafc;
        --popover: #1e293b;
        --popover-foreground: #f8fafc;
        --primary: #3b82f6;
        --primary-foreground: #f8fafc;
        --secondary: #1e293b;
        --secondary-foreground: #f8fafc;
        --muted: #475569;
        --muted-foreground: #94a3b8;
        --accent: #1e293b;
        --accent-foreground: #f8fafc;
        --destructive: #ef4444;
        --destructive-foreground: #f8fafc;
        --border: #1e293b;
        --input: #1e293b;
        --ring: #3b82f6;
    }

    .container {
        max-width: 1200px;
        margin: auto;
        padding: 2rem;
    }

    .gradio-container {
        font-family: 'Inter', -apple-system, BlinkMacSystemFont, 'Segoe UI', Roboto, sans-serif;
        background: var(--background) !important;
        color: var(--foreground) !important;
    }

    .title {
        text-align: center;
        margin-bottom: 2rem;
        font-weight: 800;
        font-size: 2.5rem;
        background: linear-gradient(to right, #3b82f6, #60a5fa);
        -webkit-background-clip: text;
        -webkit-text-fill-color: transparent;
    }

    .subtitle {
        text-align: center;
        color: var(--muted-foreground);
        margin-bottom: 2rem;
        font-size: 1.1rem;
    }

    .file-upload {
        border: 2px dashed var(--border);
        padding: 2rem;
        border-radius: 0.75rem;
        background: var(--card);
        transition: all 0.2s ease;
    }

    .file-upload:hover {
        border-color: var(--primary);
        background: var(--secondary);
    }

    .question-box {
        background: var(--card);
        padding: 1.5rem;
        border-radius: 0.75rem;
        border: 1px solid var(--border);
        transition: all 0.2s ease;
    }

    .question-box:focus-within {
        border-color: var(--primary);
        box-shadow: 0 0 0 2px rgba(59, 130, 246, 0.2);
    }

    .output-box {
        background: var(--card);
        padding: 1.5rem;
        border-radius: 0.75rem;
        border: 1px solid var(--border);
        box-shadow: 0 4px 6px -1px rgba(0, 0, 0, 0.1), 0 2px 4px -1px rgba(0, 0, 0, 0.06);
    }

    .section-title {
        font-size: 1.25rem;
        font-weight: 600;
        color: var(--foreground);
        margin-bottom: 1rem;
        display: flex;
        align-items: center;
        gap: 0.5rem;
    }

    .example-questions {
        background: var(--card);
        padding: 1.5rem;
        border-radius: 0.75rem;
        border: 1px solid var(--border);
    }

    .example-questions ul {
        list-style: none;
        padding: 0;
        margin: 0;
    }

    .example-questions li {
        padding: 0.5rem 0;
        color: var(--muted-foreground);
        display: flex;
        align-items: center;
        gap: 0.5rem;
    }

    .example-questions li::before {
        content: "•";
        color: var(--primary);
    }

    .checkbox {
        background: var(--card);
        border: 1px solid var(--border);
        border-radius: 0.5rem;
        padding: 0.5rem;
    }

    .checkbox:hover {
        border-color: var(--primary);
    }

    .button {
        background: var(--primary);
        color: var(--primary-foreground);
        border-radius: 0.5rem;
        padding: 0.75rem 1.5rem;
        font-weight: 500;
        transition: all 0.2s ease;
    }

    .button:hover {
        background: #2563eb;
        transform: translateY(-1px);
    }

    .textbox {
        background: var(--card);
        border: 1px solid var(--border);
        border-radius: 0.5rem;
        color: var(--foreground);
    }

    .textbox:focus {
        border-color: var(--primary);
        box-shadow: 0 0 0 2px rgba(59, 130, 246, 0.2);
    }
    """
    
    with gr.Blocks(css=custom_css, theme=gr.themes.Soft()) as demo:
        gr.Markdown("""
        # CSV Data Analysis Assistant
        ### Ask questions about your data and get instant insights with visualizations
        """, elem_classes="title")
        
        with gr.Row():
            with gr.Column(scale=1):
                # File upload component with better styling
                gr.Markdown("### 📁 Upload Your Data", elem_classes="section-title")
                file_input = gr.File(
                    label="Upload CSV File", 
                    file_types=[".csv"],
                    elem_classes="file-upload"
                )
                file_info = gr.Textbox(
                    label="Dataset Information",
                    lines=8,
                    interactive=False,
                    elem_classes="output-box"
                )
                file_input.change(app.handle_file_upload, inputs=file_input, outputs=file_info)
            
            with gr.Column(scale=1):
                # Question input with better styling
                gr.Markdown("### ❓ Ask a Question", elem_classes="section-title")
                question_input = gr.Textbox(
                    label="What would you like to know about your data?",
                    placeholder="Example: What is the average price of houses?",
                    lines=3,
                    elem_classes="question-box"
                )
                viz_checkbox = gr.Checkbox(
                    label="Include visualization",
                    value=False,
                    info="Check this to generate a relevant visualization",
                    elem_classes="checkbox"
                )
                
                # Output components with better styling
                gr.Markdown("### 💡 Answer", elem_classes="section-title")
                answer_output = gr.Textbox(
                    label="Response",
                    lines=4,
                    interactive=False,
                    elem_classes="output-box"
                )
                
                gr.Markdown("### 📈 Visualization", elem_classes="section-title")
                graph_output = gr.Plot(
                    label="Generated Visualization",
                    elem_classes="output-box"
                )
                
                # Submit button with better styling
                submit_button = gr.Button(
                    "Get Answer",
                    variant="primary",
                    size="large",
                    elem_classes="button"
                )
                submit_button.click(
                    app.handle_question, 
                    inputs=[question_input, viz_checkbox], 
                    outputs=[answer_output, graph_output]
                )
        
        # Add some example questions with better styling
        gr.Markdown("### 💡 Example Questions", elem_classes="section-title")
        gr.Markdown("""
        <div class="example-questions">
            <ul>
                <li>What is the total number of records in the dataset?</li>
                <li>What are the column names in the dataset?</li>
                <li>What is the average value of [column name]?</li>
                <li>Show me the distribution of [categorical column]</li>
                <li>What is the relationship between [column1] and [column2]?</li>
            </ul>
        </div>
        """)

    return demo

# Run the Gradio app
if __name__ == "__main__":
    demo = create_interface()
    demo.launch()
