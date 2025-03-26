import os
import pandas as pd
from sqlalchemy import create_engine, exc
from sklearn.metrics.pairwise import cosine_similarity


def get_db_connection():
    """Establish optimized database connection using SQLAlchemy with connection pooling."""
    try:
        db_user = os.getenv("DB_USER")
        db_password = os.getenv("DB_PASSWORD")
        db_host = os.getenv("DB_HOST")
        db_name = "tripglide"

        if not all([db_user, db_password, db_host]):
            raise ValueError("Missing database credentials in environment variables.")

        encoded_password = db_password.replace("@", "%40")
        connection_string = f"mysql+mysqlconnector://{db_user}:{encoded_password}@{db_host}/{db_name}"
        engine = create_engine(connection_string, pool_size=5, max_overflow=10)  # Optimized pooling
        return engine
    except exc.SQLAlchemyError as e:
        print(f"Database connection error: {e}")
        return None

def load_data():
    """Fetch data from the TripGlide database with enhanced error handling."""
    engine = get_db_connection()
    if engine is None:
        return None, None

    try:
        query_car = "SELECT * FROM car"
        query_rental = "SELECT * FROM rentals"

        car_table = pd.read_sql(query_car, engine)
        rental_info = pd.read_sql(query_rental, engine)

        if car_table.empty or rental_info.empty:
            print("⚠️ Warning: One or both tables are empty. Recommendations may be limited.")
            return None, None

        return car_table, rental_info
    except exc.SQLAlchemyError as e:
        print(f"Error loading data: {e}")
        return None, None

def merge_data(rental_info, car_table):
    """Merge rental info with car details, handling missing values gracefully."""
    merged_data = rental_info.merge(car_table, on='CarID', how='left')

    # Fill missing values with defaults
    merged_data.fillna({
        'Make': 'Unknown',
        'Model': 'Unknown',
        'CarType': 'Unknown',
        'Fuel_Policy': 'Same to Same',
        'Transmission': 'Manual',
        'Price_Per_Hour': 0,
        'Rating': 0,
        'Mileage_kmpl': 0,
        'Occupancy': 'Unknown',
        'AC': 'Yes',
        'Luggage_Capacity': 'Unknown',
        'Agency_Name': 'Unknown',
        'Base_Fare': 0,
        'Unlimited_Mileage':'Yes'
    }, inplace=True)

    return merged_data

def create_user_car_matrix(merged_data, selected_location):
    """Create a user-car matrix for collaborative filtering, ensuring data availability."""
    valid_locations = merged_data["Pickup_Location"].str.lower().unique()
    
    if selected_location.lower() not in valid_locations:
        print(f"❌ Invalid Pickup Location: {selected_location}. No recommendations available.")
        exit()
    
    selected_location = next(city for city in merged_data["Pickup_Location"].unique() if city.lower() == selected_location.lower())
    
    filtered_data = merged_data[merged_data['Pickup_Location'] == selected_location]

    if filtered_data.empty:
        print("❌ No rental data found for the selected location. Stopping process.")
        exit()

    return filtered_data.pivot_table(index='UserID', columns='CarID', values='TravelCode', aggfunc='count').fillna(0)

def compute_similarity(user_car_matrix):
    """Compute item-based similarity matrix, ensuring valid input."""
    if user_car_matrix is None or user_car_matrix.empty:
        print("⚠️ User-car matrix is empty. Cannot compute similarity.")
        return None

    item_similarity = cosine_similarity(user_car_matrix.T)
    return pd.DataFrame(item_similarity, index=user_car_matrix.columns, columns=user_car_matrix.columns)

def recommend_cars(user_id, user_car_matrix, item_sim_df):
    """Recommend cars based on collaborative filtering with fallback options."""
    if user_id not in user_car_matrix.index:
        print(f"⚠️ User {user_id} not found in the rental data. Providing general recommendations.")
        return user_car_matrix.columns.tolist()[:5]  # Return top 5 most rented cars as fallback

    rented_cars = user_car_matrix.loc[user_id]
    rented_cars = rented_cars[rented_cars > 0].index.tolist()
    recommended_cars = []

    for car in rented_cars:
        if car in item_sim_df.columns:
            similar_cars = item_sim_df[car].sort_values(ascending=False)[1:6]
            recommended_cars.extend(similar_cars.index.tolist())

    return list(set(recommended_cars) - set(rented_cars)) if recommended_cars else user_car_matrix.columns.tolist()[:5]

def display_recommendations(user_id, selected_location, merged_data, recommended_cars, car_table):
    """Display recommended cars in a structured format, ensuring data availability."""
    recommended_car_details = car_table[car_table["CarID"].isin(recommended_cars)]

    if recommended_car_details.empty:
        print("⚠️ No recommended car details found. Suggesting alternative popular cars.")
        # recommended_car_details = car_table.nlargest(5, 'Rating')  # Recommend top-rated cars as fallback
        exit()
    print(f"\n🚗 Recommended Cars for User {user_id} at {selected_location}\n")
    
    for _, row in recommended_car_details.iterrows():
        print(f"CarId               {row['CarID']}")
        print(f"Model               {row['Model']}")
        print(f"CarType             {row['CarType']}")
        print(f"Fuel_Policy         {row['Fuel_Policy']}")
        print(f"Transmission        {row['Transmission']}")
        print(f"Price_Per_Hour      {row['Price_Per_Hour']}")
        print(f"Rating              {row['Rating']}")
        print(f"Mileage             {row['Mileage_kmpl']}")
        print(f"Seats               {row['Occupancy']}")
        print(f"AC                  {row['AC']}")            
        print(f"Luggage_Capacity    {row['Luggage_Capacity']}")
        print(f"Agency_Name         {row['Agency_Name']}")
        print(f"Agency_Price        {row['Base_Fare']}")
        print("-" * 50)

def get_valid_user_input(prompt, validation_func):
    """Helper function to get valid user input with retries."""
    while True:
        user_input = input(prompt).strip()
        if validation_func(user_input):
            return user_input
        print("⚠️ Invalid input. Please try again.")

def main(user_id=None):
    """Robust main function handling all aspects of the recommendation system."""
    try:
        car_table, rental_info = load_data()
        if car_table is None or rental_info is None:
            print("❌ Exiting program due to data loading failure.")
            return

        merged_data = merge_data(rental_info, car_table)

        selected_location = get_valid_user_input("Enter the Pickup_location: ", lambda x: len(x) > 0)

        user_car_matrix = create_user_car_matrix(merged_data, selected_location)
        if user_car_matrix is None:
            return

        item_sim_df = compute_similarity(user_car_matrix)
        if item_sim_df is None:
            return

        if user_id is None:
            user_id=get_valid_user_input("Enter the user_id (for whom you want the recommendation): ", lambda x: x.isdigit())
            user_id=int(user_id)

        recommended_cars = recommend_cars(user_id, user_car_matrix, item_sim_df)
        display_recommendations(user_id, selected_location, merged_data, recommended_cars, car_table)

    except Exception as e:
        print(f"❌ An unexpected error occurred: {e}")

if __name__ == "__main__":
    main()