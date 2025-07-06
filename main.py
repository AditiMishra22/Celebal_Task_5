from flask import Flask, request, render_template, jsonify
import numpy as np
import pandas as pd
import joblib
from scipy import stats
import os

app = Flask(__name__)

# Load the model and scaler
model_path = os.path.join(os.path.dirname(__file__), 'model.pkl')
scaler_path = os.path.join(os.path.dirname(__file__), 'scaler.pkl')

model = joblib.load(model_path)
scaler = joblib.load(scaler_path)

@app.route('/')
def home():
    return render_template('index.html')

def engineer_features(data):
    """Apply the same feature engineering as in the notebook"""
    # Create basic features
    data['TotalSF'] = data['TotalBsmtSF'] + data['1stFlrSF'] + data['2ndFlrSF']
    data['HouseAge'] = data['YrSold'] - data['YearBuilt']
    data['Remodeled'] = (data['YearBuilt'] != data['YearRemodAdd']).astype(int)
    
    # Additional features
    data['TotalBaths'] = data['FullBath'] + 0.5 * data['HalfBath'] + data['BsmtFullBath'] + 0.5 * data['BsmtHalfBath']
    
    porch_cols = ['OpenPorchSF', 'EnclosedPorch', '3SsnPorch', 'ScreenPorch', 'WoodDeckSF']
    data['TotalPorchSF'] = data[porch_cols].sum(axis=1)
    
    data['OverallQualCond'] = data['OverallQual'] * data['OverallCond']
    
    # Basement quality score if present
    if 'BsmtQual' in data.columns:
        data['BsmtQual_num'] = data['BsmtQual'].map(
            {'Ex': 5, 'Gd': 4, 'TA': 3, 'Fa': 2, 'Po': 1}
        ).fillna(0)
    
    data['GarageAreaPerCar'] = data['GarageArea'] / (data['GarageCars'] + 1)
    data['LivingAreaRatio'] = data['GrLivArea'] / data['TotalSF']
    
    return data

def transform_features(data):
    """Apply the same transformations as in the notebook"""
    # Get numeric features
    numeric_feats = data.select_dtypes(include=['int64', 'float64']).columns
    
    # Log transform skewed features
    skewed_feats = data[numeric_feats].apply(lambda x: stats.skew(x.dropna())).sort_values(ascending=False)
    skewed_features = skewed_feats[abs(skewed_feats) > 0.75].index
    
    for feat in skewed_features:
        data[feat] = np.log1p(data[feat])
    
    # One-hot encoding for categorical variables
    data = pd.get_dummies(data, drop_first=True)
    
    return data

@app.route('/predict', methods=['POST'])
def predict():
    try:
        # Get data from form or JSON
        if request.is_json:
            input_data = request.json
            input_df = pd.DataFrame(input_data, index=[0])
        else:
            input_data = request.form.to_dict()
            # Convert string values to appropriate numeric types
            for key in input_data:
                try:
                    input_data[key] = float(input_data[key])
                except ValueError:
                    pass  # Keep as string for categorical variables
            input_df = pd.DataFrame(input_data, index=[0])
        
        # Apply feature engineering
        input_df = engineer_features(input_df)
        
        # Transform features
        input_df = transform_features(input_df)
        
        # Make sure the input data has the same columns as training data
        # This would require saving the column names from training data
        # For now, we assume the input has all required features
        
        # Scale the features
        input_scaled = scaler.transform(input_df)
        
        # Make prediction (in log scale)
        prediction_log = model.predict(input_scaled)[0]
        
        # Transform back from log scale
        prediction = np.expm1(prediction_log)
        
        return jsonify({
            'status': 'success',
            'predicted_price': round(prediction, 2),
            'formatted_price': f'${prediction:,.2f}'
        })
        
    except Exception as e:
        return jsonify({
            'status': 'error',
            'message': str(e)
        }), 400

# Route to explain the model
@app.route('/about')
def about():
    return render_template('about.html')

# Create a simple API documentation page
@app.route('/api-docs')
def api_docs():
    return render_template('api_docs.html')

if __name__ == '__main__':
    # Create templates folder if it doesn't exist
    if not os.path.exists('templates'):
        os.makedirs('templates')
    
    # Create a basic index.html template if it doesn't exist
    index_path = os.path.join('templates', 'index.html')
    
    app.run(debug=True)