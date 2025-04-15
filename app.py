from flask import Flask, render_template, request
import pandas as pd
from sklearn.preprocessing import StandardScaler
from sklearn.neighbors import KNeighborsClassifier
from sklearn.model_selection import train_test_split
import matplotlib.pyplot as plt
from io import BytesIO
import base64
import numpy as np

app = Flask(__name__)

# Load dataset
df = pd.read_csv("diabetes_data.csv")
df.fillna(df.mean(numeric_only=True), inplace=True)

# Features and target
X = df.drop('HighBP', axis=1)
y = df['HighBP']

# Feature scaling
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)

# Train-test split
X_train, _, y_train, _ = train_test_split(X_scaled, y, test_size=0.2, random_state=42)

# Train model
knn = KNeighborsClassifier(n_neighbors=5)
knn.fit(X_train, y_train)

@app.route('/')
def home():
    return render_template('index.html')

@app.route('/predict', methods=['POST'])
def predict():
    try:
        features = [
            float(request.form.get('age', 0)),
            float(request.form.get('sex', 0)),
            float(request.form.get('high_chol', 0)),
            float(request.form.get('chol_check', 0)),
            float(request.form.get('bmi', 0)),
            float(request.form.get('smoker', 0)),
            float(request.form.get('heart_disease', 0)),
            float(request.form.get('phys_activity', 0)),
            float(request.form.get('fruits', 0)),
            float(request.form.get('veggies', 0)),
            float(request.form.get('heavy_alcohol', 0)),
            float(request.form.get('gen_health', 0)),
            float(request.form.get('ment_health', 0)),
            float(request.form.get('phys_health', 0)),
            float(request.form.get('diff_walk', 0)),
            float(request.form.get('stroke', 0)),
            float(request.form.get('diabetes', 0))
        ]

        # Reshape and scale
        user_input = np.array(features).reshape(1, -1)
        scaled_input = scaler.transform(user_input)

        prediction = knn.predict(scaled_input)[0]
        prob = knn.predict_proba(scaled_input)[0]

        result = "You Have Hyper Tension" if prediction == 1 else "You Don't Have Hyper Tension"

        # Bar plot for prediction probabilities
        fig, ax = plt.subplots()
        ax.barh(['No HighBP', 'HighBP'], prob)
        ax.set_title('Probability of High Blood Pressure')
        ax.set_xlabel('Probability')
        plt.tight_layout()
        plt.close(fig)

        # Convert plot to base64
        buf = BytesIO()
        fig.savefig(buf, format='png')
        buf.seek(0)
        plot_data = base64.b64encode(buf.read()).decode('utf-8')

        return render_template('result.html', result=result, plot_data=plot_data)

    except Exception as e:
        return f"An error occurred: {e}"

if __name__ == '__main__':
    app.run(debug=True, port=8000)
