from flask import Flask, request, jsonify, render_template
from pyngrok import ngrok
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import accuracy_score

app = Flask(__name__)

ngrok_tunnel = ngrok.connect(5000)
print(f" * ngrok tunnel 'http://127.0.0.1:5000' -> {ngrok_tunnel} ")

url = 'https://raw.githubusercontent.com/jbrownlee/Datasets/master/pima-indians-diabetes.data.csv'
df = pd.read_csv(url, header=None)
df.columns = ['Pregnancies', 'Glucose', 'BloodPressure', 'SkinThickness', 'Insulin', 'BMI', 'DiabetesPedigreeFunction', 'Age', 'Outcome']

X = df.drop(columns=['Outcome'])
y = df['Outcome']

scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)

X_train, X_test, y_train, y_test = train_test_split(X_scaled, y, test_size=0.2, random_state=42, stratify=y)

model = RandomForestClassifier(random_state=42)
model.fit(X_train, y_train)

y_pred = model.predict(X_test)
accuracy = accuracy_score(y_test, y_pred)
print(f'Acurácia do modelo: {accuracy * 100:.2f}%')

@app.route('/')
def index():
    return render_template('index.html')

@app.route('/predict', methods=['POST'])
def predict():
    data = request.get_json()

    pregnancies = data['Pregnancies']
    glucose = data['Glucose']
    blood_pressure = data['BloodPressure']
    skin_thickness = data['SkinThickness']
    insulin = data['Insulin']
    bmi = data['BMI']
    diabetes_pedigree = data['DiabetesPedigreeFunction']
    age = data['Age']

    input_data = pd.DataFrame([[pregnancies, glucose, blood_pressure, skin_thickness, insulin, bmi, diabetes_pedigree, age]],
                              columns=['Pregnancies', 'Glucose', 'BloodPressure', 'SkinThickness', 'Insulin', 'BMI', 'DiabetesPedigreeFunction', 'Age'])

    input_data_scaled = scaler.transform(input_data)

    prediction = model.predict(input_data_scaled)
    prediction_prob = model.predict_proba(input_data_scaled)[0][1]  # Probabilidade de diabetes (classe 1)

    return jsonify({'prediction': int(prediction[0]), 'probability': prediction_prob * 100})

if __name__ == '__main__':
    app.run(port=5000)
