from enum import Enum, auto
from dataclasses import dataclass
from random import gauss, randint, shuffle
from typing import Any, List, Dict
from .houses import House, QualityScore
from .house_market import HousingMarket
from .consumers import Consumer, Segment


class CleaningMarketMechanism(Enum):
    INCOME_ORDER_DESCENDANT = auto()
    INCOME_ORDER_ASCENDANT = auto()
    RANDOM = auto()


@dataclass
class AnnualIncomeStatistics:
    minimum: float
    average: float
    standard_deviation: float
    maximum: float


@dataclass
class ChildrenRange:
    minimum: int = 0
    maximum: int = 5


@dataclass
class Simulation:
    housing_market_data: List[Dict[str, Any]]
    consumers_number: int
    years: int
    annual_income: AnnualIncomeStatistics
    children_range: ChildrenRange
    cleaning_market_mechanism: CleaningMarketMechanism
    down_payment_percentage: float = 0.2
    saving_rate: float = 0.3
    interest_rate: float = 0.05
    verbose: bool = False  # Added for verbosity control

    def __post_init__(self):
        self.housing_market: HousingMarket = self.create_housing_market()
        self.consumers: List[Consumer] = []

    def create_housing_market(self) -> HousingMarket:
        """
        Initialize the housing market with houses.
        """
        houses = []
        for house_data in self.housing_market_data:
            house = House(
                id=house_data["id"],
                price=house_data["price"],
                area=house_data["area"],
                bedrooms=house_data["bedrooms"],
                year_built=house_data["year_built"],
                quality_score=QualityScore(house_data["quality_score"]),
            )
            houses.append(house)
        if self.verbose:
            print(f"Housing market created with {len(houses)} houses.")
        return HousingMarket(houses)

    def create_consumers(self) -> None:
        """
        Generate a population of consumers.
        """
        self.consumers = []
        for i in range(self.consumers_number):
            income = None
            while income is None or income < self.annual_income.minimum or income > self.annual_income.maximum:
                income = gauss(self.annual_income.average, self.annual_income.standard_deviation)
            
            children_number = randint(self.children_range.minimum, self.children_range.maximum)
            segment = Segment(randint(1, len(Segment)))

            consumer = Consumer(
                id=i,
                annual_income=income,
                children_number=children_number,
                segment=segment,
                saving_rate=self.saving_rate,
                interest_rate=self.interest_rate
            )
            self.consumers.append(consumer)
        if self.verbose:
            print(f"{len(self.consumers)} consumers created.")

    def compute_consumers_savings(self) -> None:
        """
        Calculate savings for all consumers over the simulation years.
        """
        for consumer in self.consumers:
            consumer.compute_savings(self.years)
            if self.verbose:
                print(f"Consumer {consumer.id}: Savings after {self.years} years = {consumer.savings}")

    def clean_the_market(self) -> None:
        """
        Execute market transactions based on the cleaning mechanism.
        """
        if self.cleaning_market_mechanism == CleaningMarketMechanism.INCOME_ORDER_DESCENDANT:
            self.consumers.sort(key=lambda c: c.annual_income, reverse=True)
        elif self.cleaning_market_mechanism == CleaningMarketMechanism.INCOME_ORDER_ASCENDANT:
            self.consumers.sort(key=lambda c: c.annual_income)
        elif self.cleaning_market_mechanism == CleaningMarketMechanism.RANDOM:
            shuffle(self.consumers)

        if self.verbose:
            print("Cleaning the market...")
        for consumer in self.consumers:
            if self.verbose:
                print(f"Consumer {consumer.id} (income: {consumer.annual_income}) is attempting to buy a house.")
            consumer.buy_a_house(self.housing_market)

        if self.verbose:
            for house in self.housing_market.houses:
                print(f"House {house.id}: Price = {house.price}, Available = {house.available}")

    def compute_owners_population_rate(self) -> float:
        """
        Compute the percentage of consumers who own houses.
        """
        owners = sum(1 for consumer in self.consumers if consumer.house is not None)
        rate = owners / len(self.consumers) * 100
        return rate

    def compute_houses_availability_rate(self) -> float:
        """
        Compute the percentage of houses still available.
        """
        available_houses = sum(1 for house in self.housing_market.houses if house.available)
        rate = available_houses / len(self.housing_market.houses) * 100
        return rate

    def summarize_results(self) -> None:
        """
        Print a summary of the simulation results.
        """
        owners_rate = self.compute_owners_population_rate()
        houses_rate = self.compute_houses_availability_rate()
        print(f"Simulation Summary:")
        print(f"Owners population rate: {owners_rate}%")
        print(f"Houses availability rate: {houses_rate}%")


