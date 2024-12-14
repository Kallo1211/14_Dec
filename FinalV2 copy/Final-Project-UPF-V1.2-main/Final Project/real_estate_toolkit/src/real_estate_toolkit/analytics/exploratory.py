from typing import List, Dict
import polars as pl
import plotly.express as px
import plotly.graph_objects as go
from pathlib import Path

class MarketAnalyzer:
    def __init__(self, data_path: str):
        """
        Initialize the analyzer with data from a CSV file.

        Args:
            data_path (str): Path to the Ames Housing dataset
        """
        try:
            self.real_estate_data = pl.read_csv(data_path, null_values="NA")
            self.real_estate_clean_data = None
            self.output_dir = Path("src/real_estate_toolkit/analytics/outputs/")
            self.output_dir.mkdir(parents=True, exist_ok=True)
        except Exception as e:
            raise RuntimeError(f"Failed to read dataset: {e}")

    def clean_data(self) -> None:
        """
        Perform comprehensive data cleaning:
        - Handle missing values
        - Convert columns to appropriate data types
        """
        data = self.real_estate_data.clone()

        # Handle missing values
        for col in data.columns:
            null_count = data[col].null_count()
            if null_count / len(data) > 0.5:
                data = data.drop(col)
            elif data[col].dtype in [pl.Float64, pl.Int64]:
                data = data.with_columns(data[col].fill_null(data[col].median()).alias(col))
            else:
                data = data.with_columns(data[col].fill_null("Unknown").alias(col))

        # Ensure appropriate data types
        for col in data.columns:
            if data[col].dtype == pl.Utf8 and len(data[col].unique()) < 20:
                data = data.with_columns(data[col].cast(pl.Categorical).alias(col))

        self.real_estate_clean_data = data

    def generate_price_distribution_analysis(self) -> pl.DataFrame:
        """
        Analyze sale price distribution using clean data.
        """
        if self.real_estate_clean_data is None:
            raise RuntimeError("Data must be cleaned before analysis.")

        # Compute statistics
        sale_price = self.real_estate_clean_data["SalePrice"]
        price_stats = {
            "mean": sale_price.mean(),
            "median": sale_price.median(),
            "std_dev": sale_price.std(),
            "min": sale_price.min(),
            "max": sale_price.max(),
        }

        # Create and save histogram
        fig = px.histogram(
            self.real_estate_clean_data.to_pandas(),
            x="SalePrice",
            title="Sale Price Distribution",
            nbins=30,
        )
        fig.update_layout(bargap=0.1)
        fig.write_html(self.output_dir / "price_distribution.html")

        return pl.DataFrame(price_stats)

    def neighborhood_price_comparison(self) -> pl.DataFrame:
        """
        Create a boxplot comparing house prices across neighborhoods.
        """
        if self.real_estate_clean_data is None:
            raise RuntimeError("Data must be cleaned before analysis.")

        # Group by neighborhood and compute statistics
        neighborhood_stats = (
            self.real_estate_clean_data.group_by("Neighborhood")
            .agg([
                pl.col("SalePrice").mean().alias("mean_price"),
                pl.col("SalePrice").median().alias("median_price"),
                pl.col("SalePrice").std().alias("std_dev_price"),
            ])
        )

        # Convert Polars DataFrame to Pandas for Plotly visualization
        pandas_data = self.real_estate_clean_data.to_pandas()

        # Create and save boxplot
        fig = px.box(
            pandas_data,
            x="Neighborhood",
            y="SalePrice",
            title="Neighborhood Price Comparison",
        )
        fig.write_html(self.output_dir / "neighborhood_price_comparison.html")
        return neighborhood_stats

    def feature_correlation_heatmap(self, variables: List[str]) -> None:
        """
        Generate a correlation heatmap for selected variables.
        """
        if self.real_estate_clean_data is None:
            raise RuntimeError("Data must be cleaned before analysis.")

        # Compute correlation matrix
        corr_matrix = self.real_estate_clean_data.select(variables).to_pandas().corr()

        # Create and save heatmap
        fig = px.imshow(
            corr_matrix,
            text_auto=True,
            color_continuous_scale="Viridis",
            title="Feature Correlation Heatmap",
        )
        fig.write_html(self.output_dir / "correlation_heatmap.html")

    def create_scatter_plots(self) -> Dict[str, go.Figure]:
        """
        Create scatter plots exploring relationships between key features.
        """
        if self.real_estate_clean_data is None:
            raise RuntimeError("Data must be cleaned before analysis.")

        scatter_plots = {}
        plots = {
            "price_vs_sqft": ("GrLivArea", "SalePrice"),
            "price_vs_year_built": ("YearBuilt", "SalePrice"),
            "quality_vs_price": ("OverallQual", "SalePrice"),
        }

        for name, (x, y) in plots.items():
            fig = px.scatter(
                self.real_estate_clean_data.to_pandas(),
                x=x,
                y=y,
                color="Neighborhood",
                title=f"{y} vs. {x}",
                trendline="ols",
                labels={x: x, y: y},
            )
            fig.write_html(self.output_dir / f"{name}.html")
            scatter_plots[name] = fig

        return scatter_plots

if __name__ == "__main__":
    # Provide the correct path to the dataset
    data_path = "/Users/konstischumacher/Downloads/UPF/Last semester/untitled folder/FinalV2 copy/Final-Project-UPF-V1.2-main/Final Project/real_estate_toolkit/src/real_estate_toolkit/data/data_files/train.csv"
    analyzer = MarketAnalyzer(data_path)
    analyzer.clean_data()
    analyzer.generate_price_distribution_analysis()
    analyzer.neighborhood_price_comparison()
    analyzer.feature_correlation_heatmap(["SalePrice", "GrLivArea", "OverallQual", "YearBuilt"])
    analyzer.create_scatter_plots()
