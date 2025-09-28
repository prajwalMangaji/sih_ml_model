import pandas as pd
import joblib

# Load models
rf_delay = joblib.load('data/ml_data/rf_delay_model.pkl')
rf_disruption = joblib.load('data/ml_data/rf_disruption_model.pkl')
rf_occupancy = joblib.load('data/ml_data/rf_occupancy_model.pkl')

def predict_and_get_constraints(sim_input):
    """
    Predicts delay, disruption, occupancy and returns MILP constraints.
    Input: Dictionary with train_id, type, priority, departure, arrival, speed_kmh,
           direction, default_track, route_congestion, number_of_tracks,
           current_occupancy, current_occupancy_track1, current_occupancy_track2
    Output: Dictionary with predictions and MILP constraints
    """
    # Prepare input for ML (match X_test columns)
    ml_input = pd.DataFrame([[
        sim_input['type'],
        sim_input['priority'],
        sim_input['departure'],
        sim_input['arrival'],
        sim_input['speed_kmh'],
        sim_input['direction'],
        sim_input['default_track'],
        sim_input['route_congestion'],
        sim_input['number_of_tracks'],
        sim_input['current_occupancy']
    ]], columns=['type', 'priority', 'departure', 'arrival', 'speed_kmh', 'direction',
                 'default_track', 'route_congestion', 'number_of_tracks', 'current_occupancy'])

    # Predict
    delay_pred = rf_delay.predict(ml_input)[0]
    disruption_pred = rf_disruption.predict(ml_input)[0]
    occupancy_pred = rf_occupancy.predict(ml_input)[0]

    # Format MILP constraints
    milp_constraints = {
        'train_id': sim_input['train_id'],
        'predicted_delay_min': round(delay_pred, 2),
        'predicted_disruption_chance': round(disruption_pred, 2),
        'predicted_occupancy': {
            'track1': round(occupancy_pred, 2),
            'track2': round(occupancy_pred, 2)
        },
        'constraints': [
            f"Avoid track1 if occupancy > 0.5 (current: {sim_input['current_occupancy_track1']}, predicted: {round(occupancy_pred, 2)})",
            f"Avoid track2 if occupancy > 0.5 (current: {sim_input['current_occupancy_track2']}, predicted: {round(occupancy_pred, 2)})",
            f"Minimize delay (predicted: {round(delay_pred, 2)} min)",
            f"Minimize disruption (predicted: {round(disruption_pred, 2)})"
        ]
    }

    return milp_constraints