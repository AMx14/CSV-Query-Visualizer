import pandas as pd

class CSVHandler:
    """Module responsible for CSV file operations"""
    
    @staticmethod
    def load_csv(file_path: str) -> tuple[pd.DataFrame, str]:
        """Load and validate CSV file, return dataframe and info string"""
        try:
            df = pd.read_csv(file_path)
            
            # Generate preview information
            preview = f"CSV loaded successfully. Shape: {df.shape}\n\nPreview:\n{df.head(5).to_string()}\n\n"
            
            # Add column information
            preview += "Column Information:\n"
            for col in df.columns:
                dtype = df[col].dtype
                if pd.api.types.is_numeric_dtype(dtype):
                    preview += f"- {col} (numeric): min={df[col].min()}, max={df[col].max()}, mean={df[col].mean():.2f}\n"
                else:
                    preview += f"- {col} (non-numeric): {len(df[col].unique())} unique values\n"
            
            return df, preview
            
        except Exception as e:
            raise ValueError(f"Error loading CSV: {str(e)}") 