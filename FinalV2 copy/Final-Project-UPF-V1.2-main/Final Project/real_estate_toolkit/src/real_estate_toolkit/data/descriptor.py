from dataclasses import dataclass
from typing import Dict, List, Tuple, Union
import statistics
import numpy as np


@dataclass
class Descriptor:
    """Class for describing real estate data."""
    data: List[Dict[str, Union[str, float, None]]]

    def none_ratio(self, columns: List[str] = "all") -> Dict[str, float]:
        """Compute the ratio of None values per column."""
        if not self.data:
            raise ValueError("The dataset is empty.")
        
        if columns == "all":
            columns = list(self.data[0].keys())
        
        ratios = {}
        for column in columns:
            if column not in self.data[0]:
                raise ValueError(f"Column '{column}' does not exist in the data.")
            
            total = len(self.data)
            none_count = sum(1 for row in self.data if row.get(column) is None)
            ratios[column] = none_count / total if total > 0 else 0.0
            
        return ratios

    def average(self, columns: List[str] = "all") -> Dict[str, float]:
        """Compute the average value for numeric columns."""
        if not self.data:
            raise ValueError("The dataset is empty.")
        
        if columns == "all":
            columns = [col for col in self.data[0].keys() if isinstance(self.data[0].get(col), (int, float))]
        
        averages = {}
        for column in columns:
            if column not in self.data[0]:
                raise ValueError(f"Column '{column}' does not exist in the data.")
            
            values = [row[column] for row in self.data if isinstance(row.get(column), (int, float))]
            if not values:
                raise ValueError(f"No valid numeric values found in column '{column}'.")
            
            averages[column] = sum(values) / len(values)
        
        return averages

    def median(self, columns: List[str] = "all") -> Dict[str, float]:
        """Compute the median value for numeric columns."""
        if not self.data:
            raise ValueError("The dataset is empty.")
        
        if columns == "all":
            columns = [col for col in self.data[0].keys() if isinstance(self.data[0].get(col), (int, float))]
        
        medians = {}
        for column in columns:
            if column not in self.data[0]:
                raise ValueError(f"Column '{column}' does not exist in the data.")
            
            values = [row[column] for row in self.data if isinstance(row.get(column), (int, float))]
            if not values:
                raise ValueError(f"No valid numeric values found in column '{column}'.")
            
            medians[column] = statistics.median(values)
        
        return medians

    def percentile(self, columns: List[str] = "all", percentile: int = 50) -> Dict[str, float]:
        """Compute the percentile value for numeric columns."""
        if not self.data:
            raise ValueError("The dataset is empty.")
        
        if not (0 <= percentile <= 100):
            raise ValueError("Percentile must be between 0 and 100.")
        
        if columns == "all":
            columns = [col for col in self.data[0].keys() if isinstance(self.data[0].get(col), (int, float))]
        
        percentiles = {}
        for column in columns:
            if column not in self.data[0]:
                raise ValueError(f"Column '{column}' does not exist in the data.")
            
            values = [row[column] for row in self.data if isinstance(row.get(column), (int, float))]
            if not values:
                raise ValueError(f"No valid numeric values found in column '{column}'.")
            
            percentiles[column] = statistics.quantiles(values, n=100)[percentile - 1]
        
        return percentiles

    def type_and_mode(self, columns: List[str] = "all") -> Dict[str, Union[Tuple[str, float], Tuple[str, str]]]:
        """Compute the mode for variables and return their type."""
        if not self.data:
            raise ValueError("The dataset is empty.")
        
        if columns == "all":
            columns = list(self.data[0].keys())
        
        types_and_modes = {}
        for column in columns:
            if column not in self.data[0]:
                raise ValueError(f"Column '{column}' does not exist in the data.")
            
            values = [row[column] for row in self.data if row.get(column) is not None]
            if not values:
                types_and_modes[column] = ("unknown", None)
                continue
            
            if all(isinstance(val, (int, float)) for val in values):
                var_type = "numeric"
            else:
                var_type = "categorical"
            
            mode = statistics.mode(values)
            types_and_modes[column] = (var_type, mode)
        
        return types_and_modes


@dataclass
class DescriptorNumpy:
    """Class for describing real estate data using NumPy."""
    data: np.ndarray
    column_names: List[str]

    def none_ratio(self, columns: List[str] = "all") -> Dict[str, float]:
        if columns == "all":
            columns = self.column_names

        ratios = {}
        for column in columns:
            col_index = self.column_names.index(column)
            col_data = self.data[:, col_index]
            none_count = np.sum(col_data == None)  # Count None (or NaN) values
            ratios[column] = none_count / len(col_data)
        return ratios

    def average(self, columns: List[str] = "all") -> Dict[str, float]:
        if columns == "all":
            columns = self.column_names

        averages = {}
        for column in columns:
            col_index = self.column_names.index(column)
            col_data = self.data[:, col_index].astype(float)
            col_data = col_data[~np.isnan(col_data)]  # Exclude NaN values
            averages[column] = np.mean(col_data)
        return averages

    def median(self, columns: List[str] = "all") -> Dict[str, float]:
        if columns == "all":
            columns = self.column_names

        medians = {}
        for column in columns:
            col_index = self.column_names.index(column)
            col_data = self.data[:, col_index].astype(float)
            col_data = col_data[~np.isnan(col_data)]  # Exclude NaN values
            medians[column] = np.median(col_data)
        return medians

    def percentile(self, columns: List[str] = "all", percentile: int = 50) -> Dict[str, float]:
        if columns == "all":
            columns = self.column_names

        percentiles = {}
        for column in columns:
            col_index = self.column_names.index(column)
            col_data = self.data[:, col_index].astype(float)
            col_data = col_data[~np.isnan(col_data)]  # Exclude NaN values
            percentiles[column] = np.percentile(col_data, percentile)
        return percentiles
