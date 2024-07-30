from flask import Flask, request, render_template
import joblib
import pandas as pd

# Load the trained model and encoders
model = joblib.load('random_forest_model.pkl')
encoders = joblib.load('one_hot_encoder.pkl')

app = Flask(__name__)

@app.route('/')
def home():
    return render_template('index.html')

@app.route('/predict', methods=['POST'])
def predict():
    try:
        # Get form data
        YearInUni = request.form['YearInUni']
        Major = request.form['Major']
        Subsidize = int(request.form['Subsidize'])
        LackMoneyProblem = int(request.form['LackMoneyProblem'])
        SatisfiedWithGPA = int(request.form['SatisfiedWithGPA'])
        PressureFromFrsGPA = int(request.form['PressureFromFrsGPA'])
        WorryAbWork = int(request.form['WorryAbWork'])
        SleepQuality = int(request.form['SleepQuality'])
        FearLosingLoveOnes = int(request.form['FearLosingLoveOnes'])
        NegativeThought = int(request.form['NegativeThought'])
        LossOfAppetile = int(request.form['LossOfAppetile'])
        BeAddicted = int(request.form['BeAddicted'])
        Gender = request.form['Gender']
        
        # Apply mappings
        mapping_year = {'Năm 1': 1, 'Năm 2': 2, 'Năm 3': 3, 'Năm 4': 4}
        mapping_subsidize = {'Rất không hài lòng': 1, 'Không hài lòng': 2, 'Bình thường': 3, 'Hài lòng': 4, 'Rất hài lòng': 5}
        mapping_lack_money = {'Không bao giờ': 1, 'Thỉnh thoảng': 2, 'Thường xuyên': 3, 'Luôn luôn': 4}
        mapping_pressure_gpa = {'Không áp lực': 1, 'Hơi áp lực': 2, 'Áp lực': 3, 'Rất áp lực': 4}

        # Prepare the feature vector for prediction
        features = pd.DataFrame({
            'YearInUni': [mapping_year[YearInUni]], 'Major': [Major], 'Subsidize': [Subsidize], 'LackMoneyProblem': [LackMoneyProblem],
            'SatisfiedWithGPA': [SatisfiedWithGPA], "PressureFromFrs'GPA": [PressureFromFrsGPA], 'WorryAbWork': [WorryAbWork],
            'SleepQuality': [SleepQuality], 'FearLosingLoveOnes': [FearLosingLoveOnes], 'NegativeThought': [NegativeThought],
            'LossOfAppetile': [LossOfAppetile], 'BeAddicted': [BeAddicted], 'Gender': [Gender]
        })

        # One-hot encode categorical columns using the pre-fitted encoders
        for col in ['Gender', 'Major']:
            column_data = features[[col]]
            encoder = encoders[col]
            encoded_column = encoder.transform(column_data)
            encoded_df = pd.DataFrame(encoded_column.toarray(), columns=encoder.get_feature_names_out([col]))
            features = pd.concat([features, encoded_df], axis=1)
            features.drop(columns=[col], inplace=True)

        # Predict the stress level
        prediction = model.predict(features)
        
        # Map the prediction to stress level classes
        stress_level_mapping = {1: "Không stress", 2: "Bình thường", 3: "Hơi stress", 4: "Rất stress", 5: "Cực kỳ stress"}
        predicted_stress_level = stress_level_mapping[prediction[0]]
        
        return render_template('index.html', prediction_text=f'Predicted Stress Level: {predicted_stress_level}')
    
    except Exception as e:
        return str(e)  # Convert the exception to a string

if __name__ == '__main__':
    app.run(debug=True)
