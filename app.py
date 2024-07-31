from flask import Flask, request, render_template
import joblib
import pandas as pd

# Load the trained model and encoders
model = joblib.load('random_forest_model.pkl')
encoders = joblib.load('one_hot_encoder.pkl')
kmeans = joblib.load('kmeans_model.pkl')

recommendations = {
    0: """
    <strong>Khuyến nghị:</strong>
    <ol>
        <li><strong>Tư vấn tài chính:</strong> Cung cấp các buổi tư vấn về quản lý tài chính cá nhân, hỗ trợ tìm kiếm các học bổng và hỗ trợ tài chính.</li>
        <li><strong>Hỗ trợ học tập:</strong> Tổ chức các lớp học thêm, các buổi tư vấn học thuật để cải thiện GPA.</li>
        <li><strong>Chăm sóc sức khỏe tinh thần:</strong> Cung cấp các buổi tư vấn tâm lý, khuyến khích tham gia các hoạt động thể thao và thư giãn.</li>
        <li><strong>Chương trình cải thiện giấc ngủ:</strong> Giới thiệu các chương trình cải thiện giấc ngủ, cung cấp kiến thức về tầm quan trọng của giấc ngủ và cách cải thiện chất lượng giấc ngủ.</li>
    </ol>
    """,
    1: """
    <strong>Khuyến nghị:</strong>
    <ol>
        <li><strong>Duy trì hỗ trợ tài chính:</strong> Duy trì và mở rộng các chương trình hỗ trợ tài chính hiện có để đảm bảo sinh viên không gặp khó khăn về tài chính.</li>
        <li><strong>Hội thảo học tập:</strong> Tổ chức các hội thảo, lớp học nhóm để giúp sinh viên duy trì và nâng cao GPA.</li>
        <li><strong>Quản lý stress:</strong> Cung cấp các buổi hướng dẫn kỹ thuật quản lý stress, giúp sinh viên cân bằng giữa học tập và các hoạt động cá nhân.</li>
        <li><strong>Khuyến khích lối sống lành mạnh:</strong> Khuyến khích tham gia các hoạt động thể thao, yoga, và các hoạt động giải trí lành mạnh khác.</li>
    </ol>
    """,
    2: """
    <strong>Khuyến nghị:</strong>
    <ol>
        <li><strong>Tư vấn tài chính chuyên sâu:</strong> Cung cấp các buổi tư vấn tài chính chuyên sâu để giúp sinh viên quản lý tài chính cá nhân và tìm kiếm các nguồn hỗ trợ tài chính.</li>
        <li><strong>Cố vấn học tập:</strong> Cung cấp các chương trình cố vấn học tập, hỗ trợ sinh viên trong việc quản lý thời gian và lên kế hoạch học tập hiệu quả.</li>
        <li><strong>Hỗ trợ sức khỏe tinh thần toàn diện:</strong> Cung cấp các dịch vụ tư vấn tâm lý, khuyến khích tham gia các nhóm hỗ trợ và các hoạt động xã hội tích cực.</li>
        <li><strong>Chương trình sức khỏe và thể chất:</strong> Tổ chức các chương trình thể thao, yoga, và các hoạt động nâng cao sức khỏe thể chất và tinh thần.</li>
    </ol>
    """
}

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

        # Clustering
        cluster = kmeans.predict(features)
        recommendation = recommendations[cluster[0]]
        
        # Map the prediction to stress level classes
        stress_level_mapping = {1: "Không stress", 2: "Bình thường", 3: "Hơi stress", 4: "Rất stress", 5: "Cực kỳ stress"}
        predicted_stress_level = stress_level_mapping[prediction[0]]
        
        return render_template('index.html', prediction_text=f'Predicted Stress Level: {predicted_stress_level}', recommendation_text = f'{recommendation}')
    
    except Exception as e:
        return str(e)  # Convert the exception to a string

if __name__ == '__main__':
    app.run(debug=True)
