from enum import Enum
from dataclasses import dataclass
from typing import Optional

class QualityScore(Enum):
    EXCELLENT = 5
    GOOD = 4
    AVERAGE = 3
    FAIR = 2
    POOR = 1

@dataclass
class House:
    """Class representing a house in the market."""
    id: int
    price: float
    area: float
    bedrooms: int
    year_built: int
    quality_score: Optional[QualityScore] = None
    available: bool = True

    def calculate_price_per_square_foot(self) -> float:
        """
        Calculate and return the price per square foot.

        Returns:
            float: Price per square foot (rounded to 2 decimals).
        """
        if self.area == 0:
            raise ValueError("Area of the house cannot be zero.")
        return round(self.price / self.area, 2)

    def is_new_construction(self, current_year: int = 2024) -> bool:
        """
        Determine if the house is considered new construction (< 5 years old).

        Args:
            current_year (int): Current year for comparison. Defaults to 2024.

        Returns:
            bool: True if the house is new construction, False otherwise.
        """
        return (current_year - self.year_built) < 5

    def get_quality_score(self) -> str:
        """
        Return the quality score description.

        Returns:
            str: Quality score description or "Unknown" if no score is set.
        """
        if self.quality_score:
            return self.quality_score.name
        return "Unknown"

    def sell_house(self) -> None:
        """
        Mark the house as sold by setting available to False.
        """
        self.available = False
