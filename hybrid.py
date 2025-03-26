import os
import pandas as pd
from sqlalchemy import create_engine, exc
from sqlalchemy.sql import text
import pipeline_CF
import pipeline_CB

def get_db_connection():
    """Establish database connection using SQLAlchemy."""
    try:
        db_user = os.getenv("DB_USER")
        db_password = os.getenv("DB_PASSWORD")
        db_host = os.getenv("DB_HOST")
        db_name = "tripglide"

        if not all([db_user, db_password, db_host]):
            raise ValueError("Missing database credentials in environment variables.")

        encoded_password = db_password.replace("@", "%40")
        connection_string = f"mysql+mysqlconnector://{db_user}:{encoded_password}@{db_host}/{db_name}"
        engine = create_engine(connection_string, pool_pre_ping=True)
        return engine
    except exc.SQLAlchemyError as e:
        print(f"Database connection error: {e}")
        return None

def check_user_exists(user_id):
    """Check if a user exists in the rentals table."""
    engine = get_db_connection()
    if engine is None:
        print("❌ Failed to connect to the database.")
        return False
    
    try:
        query = text("SELECT COUNT(*) FROM rentals WHERE UserID = :user_id")
        with engine.connect() as connection:
            result = connection.execute(query, {"user_id": user_id}).fetchone()[0]  # Fetch first column
            return result > 0
    except exc.SQLAlchemyError as e:
        print(f"Database query error: {e}")
        return False

def main():
    """Determine if user exists in the database and call respective pipeline."""
    user_id = input("Enter User ID: ").strip()
    if not user_id.isdigit():
        print("❌ Invalid input. Please enter a numeric User ID.")
        return
    
    user_id = int(user_id)
    
    if check_user_exists(user_id):
        # print("✅ User found in the database. Running Collaborative Filtering...")
        pipeline_CF.main(user_id)  # Call the main function of pipeline_CF
    else:
        # print("⚠️ User not found in the database. Running Content-Based Filtering...")
        pipeline_CB.main()  # Call the main function of pipeline_CB

if __name__ == "__main__":
    main()
