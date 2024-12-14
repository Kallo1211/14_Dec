from dataclasses import dataclass
from typing import Dict, List, Any
import re

@dataclass
class Cleaner:
    """Class for cleaning real estate data."""
    data: List[Dict[str, Any]]

    def rename_with_best_practices(self) -> None:
        """
        Rename the columns with best practices (e.g., snake_case, descriptive names).

        Modifies the data in place by transforming all keys in each row to snake_case.
        Handles potential duplicate column names by appending a suffix.
        """
        if not self.data:
            return  # Exit early if data is empty

        # Extract all column names from the first row
        original_columns = self.data[0].keys()

        # Create a mapping from original to snake_case names
        renamed_columns = {}
        for column in original_columns:
            # Convert column name to snake_case
            new_column = re.sub(r'([A-Z])', r'_\1', column)  # Add underscores before capital letters
            new_column = re.sub(r'[^a-zA-Z0-9]', '_', new_column)  # Replace non-alphanumeric characters
            new_column = re.sub(r'_+', '_', new_column)  # Collapse multiple underscores
            new_column = new_column.lower().strip('_')  # Convert to lowercase and strip leading/trailing underscores

            # Ensure unique column names by adding suffix if needed
            counter = 1
            unique_column = new_column
            while unique_column in renamed_columns.values():
                unique_column = f"{new_column}_{counter}"
                counter += 1

            renamed_columns[column] = unique_column

        # Update all rows with renamed columns
        for row in self.data:
            new_row = {renamed_columns[key]: value for key, value in row.items()}
            row.clear()
            row.update(new_row)

    def na_to_none(self) -> None:
        """
        Replace 'NA' with None in all values in the dataset.

        Modifies the data in place to replace 'NA' string with Python's None.
        """
        for row in self.data:
            for column, value in row.items():
                if value == "NA":
                    row[column] = None
