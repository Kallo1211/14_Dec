from enum import Enum, auto
from dataclasses import dataclass
from typing import Optional
from .houses import House
from .house_market import HousingMarket

class Segment(Enum):
    FANCY = auto()  # Prefers new construction with high quality scores
    OPTIMIZER = auto()  # Focuses on price per square foot value
    AVERAGE = auto()  # Considers average market prices

@dataclass
class Consumer:
    """Class representing a consumer in the housing market."""
    id: int
    annual_income: float
    children_number: int
    segment: Segment
    house: Optional[House] = None
    savings: float = 0.0
    saving_rate: float = 0.3
    interest_rate: float = 0.05

    def compute_savings(self, years: int) -> None:
        """
        Calculate accumulated savings over time using compound interest.

        Args:
            years (int): The number of years for savings calculation.
        """
        self.savings = self.annual_income * self.saving_rate * (
            (1 + self.interest_rate) ** years - 1
        ) / self.interest_rate

    def buy_a_house(self, housing_market: HousingMarket) -> None:
        """
        Attempt to purchase a suitable house from the housing market.

        Args:
            housing_market (HousingMarket): The housing market object.
        """
        # Calculate maximum budget based on savings
        max_budget = self.savings / 0.2  # Assuming a 20% down payment

        # Determine minimum area requirements based on family size
        min_area = 1000 + (self.children_number * 200)

        # Fetch houses that meet the consumer's requirements
        matching_houses = housing_market.get_houses_that_meet_requirements(
            max_price=max_budget,
            min_area=min_area,
            segment=self.segment.name.lower()  # Pass the segment name as lowercase
        )

        # Attempt to purchase the first available house
        if matching_houses:
            self.house = matching_houses[0]
            self.house.sell_house()
