import os
import pandas as pd
import numpy as np
from sqlalchemy import create_engine
from sqlalchemy.exc import SQLAlchemyError
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity

class CarRecommendationSystem:
    def __init__(self):
        """Initialize database connection and load data."""
        self.engine = self.connect_to_db()
        self.car_df = self.fetch_data_from_db()
        self.filtered_cars = None
        self.similarity_matrix = None

    def connect_to_db(self):
        """Establish connection to MySQL database using SQLAlchemy."""
        try:
            password = os.getenv("DB_PASSWORD", "")
            encoded_password = password.replace("@", "%40") if password else ""
            db_url = f"mysql+mysqlconnector://{os.getenv('DB_USER')}:{encoded_password}@{os.getenv('DB_HOST')}/tripglide"
            return create_engine(db_url, pool_pre_ping=True)
        except SQLAlchemyError:
            exit("Database connection failed.")

    def fetch_data_from_db(self):
        """Retrieve car rental data from the 'car' table."""
        query = "SELECT * FROM car"
        try:
            with self.engine.connect() as connection:
                return pd.read_sql(query, connection)
        except SQLAlchemyError:
            exit("Failed to fetch data.")

    def filter_by_location(self, user_city):
        """Filter cars based on user location."""
        valid_cities=self.car_df["City"].str.lower().unique()
        if user_city.lower() not in valid_cities:
            print("\n❌ Error: Invalid Pickup Location. Please enter a valid city from the database.")
            exit()
        self.filtered_cars = self.car_df[self.car_df["City"].str.lower() == user_city.lower()]

    def apply_user_preferences(self, preferred_type=None, max_price=None, ac_required=None, unlimited_mileage=None):
        """Filter cars based on user preferences and handle incorrect price inputs."""
        # Set default values if user does not enter anything
        preferred_type = preferred_type.strip() if preferred_type else "SUV"
        max_price = max_price.strip() if max_price else "1000"
        ac_required = ac_required.strip().lower() if ac_required else "yes"
        unlimited_mileage = unlimited_mileage.strip().lower() if unlimited_mileage else "yes"

        valid_types = {"suv", "sedan", "hatchback", "luxury"}
        if preferred_type.lower() not in valid_types:
            print("\n❌ Error: Invalid Car Type. Choose from SUV, Sedan, Hatchback, or Luxury.")
            return False
        
        try:
            max_price = float(max_price)
        except ValueError:
            print("\n❌ Invalid price input. Please enter a numeric value.")
            return False

        # Get the minimum price in the dataset
        min_price = self.car_df["Price_Per_Hour"].min()

        if max_price < min_price:
            print(f"\n❌ Error: No cars available under ₹{max_price}/hour. The lowest price available is ₹{min_price}/hour.")
            return False

        # Validate AC & Unlimited Mileage inputs
        if ac_required not in {"yes", "no"}:
            print("\n❌ Error: AC must be either 'Yes' or 'No'.")
            return False
        if unlimited_mileage not in {"yes", "no"}:
            print("\n❌ Error: Unlimited Mileage must be either 'Yes' or 'No'.")
            return False

        # Apply filtering
        self.filtered_cars = self.filtered_cars[
            (self.filtered_cars["CarType"].str.lower() == preferred_type.lower()) &
            (self.filtered_cars["Price_Per_Hour"] <= max_price) &
            (self.filtered_cars["AC"].str.lower() == ac_required) &
            (self.filtered_cars["Unlimited_Mileage"].str.lower() == unlimited_mileage)
        ]

        if self.filtered_cars.empty:
            print(f"\n⚠️ No cars match your preferences under ₹{max_price}/hour. Try increasing your budget.")
            return False
        return True

    def compute_similarity(self):
        """Compute cosine similarity between car features."""
        features = ["Make", "Model", "CarType", "Transmission", "Fuel_Policy"]

        # Ensure a copy to avoid `SettingWithCopyWarning`
        self.filtered_cars = self.filtered_cars.copy()

        # Fill missing values
        self.filtered_cars.loc[:, features] = self.filtered_cars[features].fillna("Unknown")
        self.filtered_cars["combined_features"] = self.filtered_cars[features].agg(" ".join, axis=1)

        vectorizer = TfidfVectorizer()
        feature_vectors = vectorizer.fit_transform(self.filtered_cars["combined_features"])
        self.similarity_matrix = cosine_similarity(feature_vectors)

    def recommend_similar_cars(self):
        """Recommend cars similar to the highest-rated car in the filtered list."""
        self.filtered_cars = self.filtered_cars.reset_index(drop=True)
        selected_car_index = self.filtered_cars["Rating"].idxmax()

        if selected_car_index >= len(self.similarity_matrix):
            return

        similarity_scores = self.similarity_matrix[selected_car_index]
        similar_car_indices = np.argsort(similarity_scores)[::-1][1:6]  # Top 5 similar cars

        print("\n🔹 Recommended Similar Cars:")
        for idx in similar_car_indices:
            car_details = self.filtered_cars.iloc[idx][["CarID","Model", "CarType", "Fuel_Policy", "Transmission", "Mileage_kmpl", "Occupancy", "Price_Per_Hour", "Agency_Name", "Base_Fare"]]
            print(car_details)
            print("-" * 50)

def main():
    """Main function to execute the car recommendation system."""
    try:
        recommender = CarRecommendationSystem()
        user_city = input("Enter Pickup_location: ").strip()
        recommender.filter_by_location(user_city)

        if not recommender.filtered_cars.empty:
            # preferred_type = input("Enter preferred Car Type (SUV, Sedan, etc.): ").strip()
            # max_price = input("Enter max price per hour (INR): ").strip()
            # ac_required = input("Do you want AC? (Yes/No): ").strip().lower()
            # unlimited_mileage = input("Do you need Unlimited Mileage? (Yes/No): ").strip().lower()
            preferred_type = input("Enter preferred Car Type (SUV, Sedan, etc.): ").strip() or None
            max_price = input("Enter max price per hour (INR):").strip() or None
            ac_required = input("Do you want AC? (Yes/No): ").strip() or None
            unlimited_mileage = input("Do you need Unlimited Mileage? (Yes/No): ").strip() or None

            # Stop execution if no cars match the filters
            if not recommender.apply_user_preferences(preferred_type, max_price, ac_required, unlimited_mileage):
                return  # Stop further execution

            recommender.compute_similarity()
            recommender.recommend_similar_cars()

    except Exception as e:
        print(f"Unexpected error: {e}")

if __name__ == "__main__":
    main()