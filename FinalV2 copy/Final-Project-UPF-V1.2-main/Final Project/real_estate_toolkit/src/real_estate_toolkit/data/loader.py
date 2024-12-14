from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List, Any, Union
import csv

@dataclass
class DataLoader:
    """Class for loading and basic processing of real estate data."""
    data_path: Union[Path, str]  # Accept either a Path or string

    def __post_init__(self):
        # Ensure data_path is a Path object
        if isinstance(self.data_path, str):
            self.data_path = Path(self.data_path)

    def load_data_from_csv(self, file_name: str = "train.csv") -> List[Dict[str, Any]]:
        """
        Load data from a specific CSV file into a list of dictionaries.

        Args:
            file_name (str): Name of the CSV file to load (default: "train.csv").
        
        Returns:
            List[Dict[str, Any]]: A list of dictionaries where each dictionary represents a row.
        """
        data = []
        file_path = self.data_path / file_name  # Combine the base path and file name

        try:
            with open(file_path, mode='r', newline='', encoding='utf-8') as file:
                reader = csv.DictReader(file)  # Automatically maps columns to values
                data = [row for row in reader]  # Convert reader object to a list of dictionaries
        except FileNotFoundError:
            raise FileNotFoundError(f"The file {file_name} does not exist at {self.data_path}.")
        except Exception as e:
            raise RuntimeError(f"An error occurred while reading {file_name}: {e}")
        
        return data

    def validate_columns(self, required_columns: List[str], file_name: str = "train.csv") -> bool:
        """
        Validate that the specified CSV file contains all required columns.

        Args:
            required_columns (List[str]): List of column names that are required.
            file_name (str): Name of the CSV file to validate (default: "train.csv").
        
        Returns:
            bool: True if all required columns are present, False otherwise.
        """
        file_path = self.data_path / file_name  # Combine the base path and file name

        try:
            with open(file_path, mode='r', newline='', encoding='utf-8') as file:
                reader = csv.reader(file)
                header = next(reader)  # Extract header from the first row
            
            return all(col in header for col in required_columns)  # Check required columns
        except FileNotFoundError:
            raise FileNotFoundError(f"The file {file_name} does not exist at {self.data_path}.")
        except Exception as e:
            raise RuntimeError(f"An error occurred while validating {file_name}: {e}")
    
    def infer_and_convert_types(self, data: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
        """
        Infer and convert column data types for numeric columns.

        Args:
            data: List of dictionaries where each dictionary represents a row.

        Returns:
            List of dictionaries with numeric values converted to appropriate types.
        """
        for row in data:
            for column, value in row.items():
                # Skip None or empty values
                if value is None or value == "":
                    continue
                try:
                    # Convert to float if it has a decimal point, else int
                    if "." in value:
                        row[column] = float(value)
                    else:
                        row[column] = int(value)
                except ValueError:
                    pass  # Leave non-numeric values as-is
        return data

if __name__ == "__main__":
    # Test functions to validate the DataLoader functionality
    def test_validate_columns():
        # Specify the path to your dataset directory
        data_path = "/Users/konstischumacher/Downloads/UPF/Last semester/untitled folder/FinalV2 copy/Final-Project-UPF-V1.2-main/Final Project/real_estate_toolkit/src/real_estate_toolkit/data/data_files"  # Replace with the actual path
        loader = DataLoader(data_path)
        
        # Test file name and required columns
        required_columns = ["Id", "SalePrice", "LotArea", "YearBuilt", "BedroomAbvGr"]
        
        try:
            is_valid = loader.validate_columns(required_columns)
            if is_valid:
                print("Validation Passed: All required columns are present.")
            else:
                print("Validation Failed: Some required columns are missing.")
        except Exception as e:
            print(f"Test failed: {e}")
    
    def test_load_data():
        data_path = "/Users/konstischumacher/Downloads/UPF/Last semester/untitled folder/FinalV2 copy/Final-Project-UPF-V1.2-main/Final Project/real_estate_toolkit/src/real_estate_toolkit/data/data_files"  # Replace with the actual path
        loader = DataLoader(data_path)
        
        try:
            data = loader.load_data_from_csv()
            print(f"Loaded {len(data)} rows from train.csv successfully.")
        except Exception as e:
            print(f"Data loading failed: {e}")

    # Run tests
    test_validate_columns()
    test_load_data()

 
   