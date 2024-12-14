from typing import List, Dict, Any
import polars as pl
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.impute import SimpleImputer
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.linear_model import LinearRegression
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import (
    mean_squared_error,
    mean_absolute_error,
    r2_score,
    mean_absolute_percentage_error
)
from pathlib import Path

class HousePricePredictor:
    def __init__(self, train_data_path: str, test_data_path: str):
        """
        Initialize the predictor class with paths to the training and testing datasets.
        
        Args:
            train_data_path (str): Path to the training dataset CSV file.
            test_data_path (str): Path to the testing dataset CSV file.
        """
        self.train_data = pl.read_csv(train_data_path, null_values="NA")
        self.test_data = pl.read_csv(test_data_path, null_values="NA")

        # Define the target column
        self.target_column = "SalePrice"

        self.pipeline = None
        self.models = {}

        # Debugging: Print columns to verify
        print("Train Data Columns:", self.train_data.columns)
        print("Test Data Columns:", self.test_data.columns)

    def clean_data(self):
        """
        Perform comprehensive data cleaning on the training and testing datasets.
        """
        # Handle missing values and ensure correct data types
        for df in [self.train_data, self.test_data]:
            for col in df.columns:
                null_count = df[col].null_count()
                if null_count / len(df) > 0.5:
                    df = df.drop(col)
                elif df[col].dtype in [pl.Float64, pl.Int64]:
                    df = df.with_columns(df[col].fill_null(df[col].median()).alias(col))
                else:
                    df = df.with_columns(df[col].fill_null("Unknown").alias(col))

        self.train_data = self.train_data.with_columns([pl.col(self.target_column).cast(pl.Float64)])
        self.test_data = self.test_data

        # Debugging: Ensure target column exists after cleaning
        if self.target_column not in self.train_data.columns:
            raise ValueError(f"Target column {self.target_column} is missing after cleaning.")

    def prepare_features(self, target_column: str = 'SalePrice', selected_predictors: List[str] = None):
        """
        Prepare the dataset for machine learning by separating features and the target variable, 
        and preprocessing them for training and testing.

        Args:
            target_column (str): Name of the target variable column. Default is 'SalePrice'.
            selected_predictors (List[str]): Specific columns to use as predictors. 
                                            If None, use all columns except the target.

        Returns:
            - X_train, X_test, y_train, y_test: Training and testing sets.
        """
        if target_column not in self.train_data.columns:
            raise ValueError(f"Target column {target_column} is missing from the training data.")

        predictors = selected_predictors or [col for col in self.train_data.columns if col != target_column]

        numeric_features = [col for col in predictors if self.train_data[col].dtype in [pl.Float64, pl.Int64]]
        categorical_features = [col for col in predictors if self.train_data[col].dtype == pl.Utf8]

        numeric_transformer = Pipeline(steps=[
            ('imputer', SimpleImputer(strategy='mean')),
            ('scaler', StandardScaler())
        ])

        categorical_transformer = Pipeline(steps=[
            ('imputer', SimpleImputer(strategy='constant', fill_value='missing')),
            ('onehot', OneHotEncoder(handle_unknown='ignore'))
        ])

        preprocessor = ColumnTransformer(
            transformers=[
                ('num', numeric_transformer, numeric_features),
                ('cat', categorical_transformer, categorical_features)
            ]
        )

        X = self.train_data[predictors].to_pandas()
        y = self.train_data[target_column].to_pandas()

        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
        self.pipeline = Pipeline(steps=[('preprocessor', preprocessor)])

        return X_train, X_test, y_train, y_test

    def train_baseline_models(self):
        """
        Train and evaluate baseline machine learning models for house price prediction.
        """
        X_train, X_test, y_train, y_test = self.prepare_features()

        models = {
            'Linear Regression': LinearRegression(),
            'Random Forest': RandomForestRegressor(random_state=42)
        }

        results = {}
        for name, model in models.items():
            pipeline = Pipeline(steps=[
                ('preprocessor', self.pipeline.named_steps['preprocessor']),
                ('model', model)
            ])

            pipeline.fit(X_train, y_train)
            y_pred_train = pipeline.predict(X_train)
            y_pred_test = pipeline.predict(X_test)

            metrics = {
                'MSE': mean_squared_error(y_test, y_pred_test),
                'MAE': mean_absolute_error(y_test, y_pred_test),
                'R2': r2_score(y_test, y_pred_test),
                'MAPE': mean_absolute_percentage_error(y_test, y_pred_test)
            }

            results[name] = {
                'metrics': metrics,
                'model': pipeline
            }

            print(f"Model: {name}, Metrics: {metrics}")

        self.models = results

    def forecast_sales_price(self, model_type: str = 'Linear Regression'):
        """
        Use the trained model to forecast house prices on the test dataset.
        """
        if model_type not in self.models:
            raise ValueError(f"Model {model_type} is not trained.")

        model = self.models[model_type]['model']

        X_test = self.test_data.to_pandas()
        predictions = model.predict(X_test)

        ids = self.test_data['Id'].to_pandas()

        # Create submission file
        submission = pl.DataFrame({"Id": ids, "SalePrice": predictions})
        output_path = Path("src/real_estate_toolkit/ml_models/outputs/submission.csv")
        output_path.parent.mkdir(parents=True, exist_ok=True)
        submission.write_csv(output_path)

if __name__ == "__main__":
    train_path = "src/real_estate_toolkit/data/data_files/train.csv"
    test_path = "src/real_estate_toolkit/data/data_files/test.csv"
    predictor = HousePricePredictor(train_path, test_path)

    # Test the clean_data method
    predictor.clean_data()

    # Test the prepare_features method
    X_train, X_test, y_train, y_test = predictor.prepare_features()
    print("Train-test split successful. Shapes:")
    print(f"X_train: {X_train.shape}, X_test: {X_test.shape}, y_train: {y_train.shape}, y_test: {y_test.shape}")

    # Train models
    predictor.train_baseline_models()

    # Forecast using Linear Regression
    predictor.forecast_sales_price("Linear Regression")
