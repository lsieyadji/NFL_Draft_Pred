from flask import Flask, render_template, request, jsonify
import joblib
import numpy as np
import os

app = Flask(__name__)

# Load stacking ensemble model
model_path = os.path.join(os.path.dirname(__file__), '..', 'models', 'stacking_ensemble_model.pkl')
model = joblib.load(model_path)

# All positions and position types for 35-feature model
positions = [
    'C', 'CB', 'DE', 'DT', 'FB', 'FS', 'ILB', 'K', 'LS', 'OG',
    'OLB', 'OT', 'P', 'QB', 'RB', 'S', 'SS', 'TE', 'WR'
]

position_types = [
    'backs_receivers', 'defensive_back', 'defensive_lineman',
    'kicking_specialist', 'line_backer', 'offensive_lineman', 'other_special'
]


@app.route('/')
def home():
    return render_template('index.html', positions=positions)


@app.route('/predict', methods=['POST'])
def predict():
    try:
        # Get form data
        data = request.json

        # Create feature array with zeros - 35 features for stacking
        features = np.zeros(35)

        # Map basic features (first 8 features)
        features[0] = float(data['height'])
        features[1] = float(data['weight'])
        features[2] = float(data['bmi'])
        features[3] = float(data['age'])
        features[4] = float(data['sprint_40yd'])
        features[5] = float(data['vertical_jump'])
        features[6] = float(data['bench_press'])
        features[7] = float(data['broad_jump'])

        # Map position (one-hot encoding) - positions start at index 8
        position_index = 8 + positions.index(data['position'])
        features[position_index] = 1

        # Map position type automatically based on position
        position_type = get_position_type(data['position'])
        position_type_index = 8 + len(positions) + position_types.index(position_type)
        features[position_type_index] = 1

        # Make prediction
        probability = model.predict_proba([features])[0][1]
        drafted = probability > 0.5

        return jsonify({
            'drafted': bool(drafted),
            'probability': round(probability * 100, 2),
            'confidence': 'high' if probability > 0.7 else 'medium' if probability > 0.5 else 'low'
        })

    except Exception as e:
        return jsonify({'error': str(e)}), 400


def get_position_type(position):
    """Map specific position to position type"""
    position_type_map = {
        'RB': 'backs_receivers', 'WR': 'backs_receivers', 'TE': 'backs_receivers', 'FB': 'backs_receivers',
        'QB': 'backs_receivers',
        'CB': 'defensive_back', 'FS': 'defensive_back', 'SS': 'defensive_back', 'S': 'defensive_back',
        'DE': 'defensive_lineman', 'DT': 'defensive_lineman',
        'OLB': 'line_backer', 'ILB': 'line_backer',
        'OT': 'offensive_lineman', 'OG': 'offensive_lineman', 'C': 'offensive_lineman',
        'K': 'kicking_specialist', 'P': 'kicking_specialist',
        'LS': 'other_special'
    }
    return position_type_map.get(position, 'backs_receivers')


if __name__ == '__main__':
    app.run(debug=True)