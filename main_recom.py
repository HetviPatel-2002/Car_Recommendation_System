from flask import Flask, request, jsonify, render_template
import pipeline_CB
import pipeline_CF
import pandas as pd  # Import Pandas to retrieve car details

# Load car data once (Assuming car_table has all car details)
car_table, _ = pipeline_CF.load_data()

def get_car_details(car_ids):
    """Retrieve full car details from car_table using car IDs"""
    return car_table[car_table["CarID"].isin(car_ids)].to_dict(orient="records")

def create_app():
    app = Flask(__name__)

    @app.route("/")
    def home():
        return render_template("index.html")

    @app.route("/recommend", methods=["GET"])
    def recommend():
        user_id = request.args.get("user_id")
        pickup_location = request.args.get("pickup_location")

        if not pickup_location:
            return jsonify({"error": "Missing pickup_location"}), 400

        try:
            if user_id and user_id.isdigit():
                user_id = int(user_id)

                # Compute required matrices
                _, rental_info = pipeline_CF.load_data()
                merged_data = pipeline_CF.merge_data(rental_info, car_table)
                user_car_matrix = pipeline_CF.create_user_car_matrix(merged_data, pickup_location)
                item_sim_df = pipeline_CF.compute_similarity(user_car_matrix)

                # Get recommended car IDs
                recommended_car_ids = pipeline_CF.recommend_cars(user_id, user_car_matrix, item_sim_df)

                if not recommended_car_ids:
                    recommended_car_ids = pipeline_CB.recommend_cars(pickup_location)
            else:
                recommended_car_ids = pipeline_CB.recommend_cars(pickup_location)

            # Fetch full car details
            recommended_cars = get_car_details(recommended_car_ids)

            return jsonify({"recommended_cars": recommended_cars})
        except Exception as e:
            return jsonify({"error": str(e)}), 500

    return app

if __name__ == "__main__":
    app = create_app()
    app.run(debug=True)
