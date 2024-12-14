from typing import List, Optional
from .houses import House

class HousingMarket:
    """Class representing the housing market."""
    
    def __init__(self, houses: List[House]):
        self.houses: List[House] = houses

    def get_house_by_id(self, house_id: int) -> Optional[House]:
        """
        Retrieve a specific house by its ID.

        Args:
            house_id (int): ID of the house to retrieve.

        Returns:
            Optional[House]: The house with the specified ID, or None if not found.
        """
        for house in self.houses:
            if house.id == house_id:
                return house
        return None

    def calculate_average_price(self, bedrooms: Optional[int] = None) -> float:
        """
        Calculate the average house price, optionally filtered by the number of bedrooms.

        Args:
            bedrooms (Optional[int]): Number of bedrooms to filter by. Defaults to None.

        Returns:
            float: The average price of the filtered houses, or all houses if no filter is applied.
        """
        filtered_houses = (
            [house for house in self.houses if house.bedrooms == bedrooms]
            if bedrooms is not None
            else self.houses
        )
        if not filtered_houses:
            raise ValueError("No houses available to calculate the average price.")
        
        total_price = sum(house.price for house in filtered_houses)
        return total_price / len(filtered_houses)

    def get_houses_that_meet_requirements(self, max_price: float, min_area: float, segment: Optional[str] = None) -> List[House]:
        """
        Filter houses based on buyer requirements, including segment-specific preferences.

        Args:
            max_price (float): Maximum price a buyer is willing to pay.
            min_area (float): Minimum area a buyer requires.
            segment (Optional[str]): Consumer segment (e.g., "fancy", "optimizer", "average"). Defaults to None.

        Returns:
            List[House]: A list of houses that meet the criteria.
        """
        matching_houses = [
            house for house in self.houses
            if house.available and house.price <= max_price and house.area >= min_area
        ]

        if segment == "fancy":
            # Fancy buyers prefer new construction and high quality
            matching_houses = [
                house for house in matching_houses
                if house.quality_score and house.quality_score.value == 5 and house.year_built >= 2020
            ]
        elif segment == "optimizer":
            # Optimizers focus on value (price per square foot)
            matching_houses.sort(key=lambda h: h.price / h.area)
        elif segment == "average":
            # Average buyers are less specific, already covered by price and area filters
            pass

        return matching_houses


