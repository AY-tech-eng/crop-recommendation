import joblib
from flask import Flask, render_template, request, jsonify
import numpy as np
import json

app = Flask(__name__)

# Load the trained model
model = joblib.load('random_forest_model.pkl')

# Load crop knowledge database
with open('crop_data.json', 'r') as f:
    crop_data = json.load(f)

def get_confidence_scores(input_data):
    """Get prediction probabilities for all crops"""
    try:
        probabilities = model.predict_proba(input_data)[0]
        classes = model.classes_
        
        # Create list of (crop, confidence) tuples
        crop_confidence = list(zip(classes, probabilities))
        
        # Sort by confidence (highest first)
        crop_confidence.sort(key=lambda x: x[1], reverse=True)
        
        return crop_confidence
    except:
        return []

def calculate_suitability_score(input_features, crop_name):
    """Calculate how well input conditions match ideal conditions for a crop"""
    if crop_name.lower() not in crop_data:
        return 0
    
    ideal = crop_data[crop_name.lower()]['ideal_conditions']
    feature_names = ['N', 'P', 'K', 'temperature', 'humidity', 'ph', 'rainfall']
    
    total_score = 0
    for i, feature in enumerate(feature_names):
        value = input_features[i]
        ideal_range = ideal[feature]
        min_val, max_val = ideal_range[0], ideal_range[1]
        
        # Check if value is within ideal range
        if min_val <= value <= max_val:
            total_score += 100
        else:
            # Calculate how far off from ideal range
            if value < min_val:
                deviation = ((min_val - value) / min_val) * 100
            else:
                deviation = ((value - max_val) / max_val) * 100
            
            # Score decreases with deviation
            score = max(0, 100 - deviation)
            total_score += score
    
    # Average score across all features
    return round(total_score / len(feature_names), 1)

def get_fertilizer_recommendation(N, P, K, crop_name):
    """Provide fertilizer adjustment recommendations"""
    recommendations = []
    
    if crop_name.lower() not in crop_data:
        return ["No specific fertilizer data available for this crop."]
    
    ideal = crop_data[crop_name.lower()]['ideal_conditions']
    
    # Check Nitrogen
    if N < ideal['N'][0]:
        deficit = ideal['N'][0] - N
        recommendations.append(f"⬆️ Nitrogen Low: Add {deficit:.0f} kg/ha of Urea or Ammonium Sulfate")
    elif N > ideal['N'][1]:
        excess = N - ideal['N'][1]
        recommendations.append(f"⬇️ Nitrogen High: Reduce by {excess:.0f} kg/ha. Consider cover crops to absorb excess")
    else:
        recommendations.append("✅ Nitrogen levels are optimal")
    
    # Check Phosphorus
    if P < ideal['P'][0]:
        deficit = ideal['P'][0] - P
        recommendations.append(f"⬆️ Phosphorus Low: Add {deficit:.0f} kg/ha of Single Super Phosphate (SSP)")
    elif P > ideal['P'][1]:
        excess = P - ideal['P'][1]
        recommendations.append(f"⬇️ Phosphorus High: Reduce by {excess:.0f} kg/ha in next season")
    else:
        recommendations.append("✅ Phosphorus levels are optimal")
    
    # Check Potassium
    if K < ideal['K'][0]:
        deficit = ideal['K'][0] - K
        recommendations.append(f"⬆️ Potassium Low: Add {deficit:.0f} kg/ha of Muriate of Potash (MOP)")
    elif K > ideal['K'][1]:
        excess = K - ideal['K'][1]
        recommendations.append(f"⬇️ Potassium High: Reduce by {excess:.0f} kg/ha in next season")
    else:
        recommendations.append("✅ Potassium levels are optimal")
    
    return recommendations

def get_rotation_suggestions(crop_name):
    """Get crop rotation recommendations"""
    if crop_name.lower() not in crop_data:
        return []
    
    compatible_crops = crop_data[crop_name.lower()]['rotation_compatible']
    return compatible_crops

def get_environmental_advice(temp, humidity, ph, rainfall, crop_name):
    """Provide advice on environmental conditions"""
    advice = []
    
    if crop_name.lower() not in crop_data:
        return ["Environmental data not available for this crop."]
    
    ideal = crop_data[crop_name.lower()]['ideal_conditions']
    
    # Temperature
    if temp < ideal['temperature'][0]:
        advice.append(f"🌡️ Temperature too low. Ideal: {ideal['temperature'][0]}-{ideal['temperature'][1]}°C. Consider mulching or greenhouse.")
    elif temp > ideal['temperature'][1]:
        advice.append(f"🌡️ Temperature too high. Ideal: {ideal['temperature'][0]}-{ideal['temperature'][1]}°C. Increase irrigation and shading.")
    else:
        advice.append(f"✅ Temperature optimal ({ideal['temperature'][0]}-{ideal['temperature'][1]}°C)")
    
    # Humidity
    if humidity < ideal['humidity'][0]:
        advice.append(f"💧 Humidity low. Ideal: {ideal['humidity'][0]}-{ideal['humidity'][1]}%. Increase irrigation frequency.")
    elif humidity > ideal['humidity'][1]:
        advice.append(f"💧 Humidity high. Ideal: {ideal['humidity'][0]}-{ideal['humidity'][1]}%. Ensure good drainage and air circulation.")
    else:
        advice.append(f"✅ Humidity optimal ({ideal['humidity'][0]}-{ideal['humidity'][1]}%)")
    
    # pH
    if ph < ideal['ph'][0]:
        diff = ideal['ph'][0] - ph
        advice.append(f"🧪 Soil too acidic (pH {ph}). Add lime to raise pH by {diff:.1f} units.")
    elif ph > ideal['ph'][1]:
        diff = ph - ideal['ph'][1]
        advice.append(f"🧪 Soil too alkaline (pH {ph}). Add sulfur or organic matter to lower pH by {diff:.1f} units.")
    else:
        advice.append(f"✅ pH optimal ({ideal['ph'][0]}-{ideal['ph'][1]})")
    
    # Rainfall
    if rainfall < ideal['rainfall'][0]:
        deficit = ideal['rainfall'][0] - rainfall
        advice.append(f"🌧️ Rainfall insufficient. Need {deficit:.0f}mm more. Set up irrigation system.")
    elif rainfall > ideal['rainfall'][1]:
        excess = rainfall - ideal['rainfall'][1]
        advice.append(f"🌧️ Rainfall excessive by {excess:.0f}mm. Ensure proper drainage to prevent waterlogging.")
    else:
        advice.append(f"✅ Rainfall optimal ({ideal['rainfall'][0]}-{ideal['rainfall'][1]}mm)")
    
    return advice

@app.route('/predict', methods=['POST'])
def predict():
    try:
        # Get form data
        N = float(request.form.get('N'))
        P = float(request.form.get('P'))
        K = float(request.form.get('K'))
        temperature = float(request.form.get('temperature'))
        humidity = float(request.form.get('humidity'))
        ph = float(request.form.get('ph'))
        rainfall = float(request.form.get('rainfall'))

        # Prepare input data
        features = [N, P, K, temperature, humidity, ph, rainfall]
        input_data = np.array([features])
        
        # Get primary prediction
        prediction = model.predict(input_data)[0]
        
        # Get confidence scores for all crops
        all_confidences = get_confidence_scores(input_data)
        
        # Get top 3 alternative crops
        top_3 = all_confidences[:3]
        alternatives = []
        for crop, conf in top_3[1:]:  # Skip the first one (main prediction)
            suitability = calculate_suitability_score(features, crop)
            alternatives.append({
                'name': crop,
                'confidence': round(conf * 100, 1),
                'suitability': suitability
            })
        
        # Calculate suitability score for main prediction
        main_suitability = calculate_suitability_score(features, prediction)
        main_confidence = round(all_confidences[0][1] * 100, 1)
        
        # Get fertilizer recommendations
        fertilizer_advice = get_fertilizer_recommendation(N, P, K, prediction)
        
        # Get crop rotation suggestions
        rotation_crops = get_rotation_suggestions(prediction)
        
        # Get environmental advice
        env_advice = get_environmental_advice(temperature, humidity, ph, rainfall, prediction)
        
        # Get crop info
        crop_info = crop_data.get(prediction.lower(), {})
        season = crop_info.get('season', 'N/A')
        growth_duration = crop_info.get('growth_duration', 'N/A')
        market_value = crop_info.get('market_value', 'N/A')

        return render_template('results.html', 
            prediction=str(prediction),
            confidence=main_confidence,
            suitability=main_suitability,
            alternatives=alternatives,
            fertilizer_advice=fertilizer_advice,
            rotation_crops=rotation_crops,
            env_advice=env_advice,
            season=season,
            growth_duration=growth_duration,
            market_value=market_value,
            N=N, P=P, K=K, 
            temperature=temperature,
            humidity=humidity, 
            ph=ph, 
            rainfall=rainfall)
            
    except Exception as e:
        return render_template('results.html', 
            prediction="Error: " + str(e),
            N=0, P=0, K=0, temperature=0,
            humidity=0, ph=0, rainfall=0)

@app.route('/')
def welcome():
    return render_template('welcome.html')

@app.route('/predict-page')
def predict_page():
    return render_template('index.html')

if __name__ == '__main__':
    app.run(debug=True)
