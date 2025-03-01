from flask import Flask, request, render_template
import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split

app = Flask(__name__)

# Load the diabetes dataset
df = pd.read_csv('diabetes.csv')

# Split the data into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(df.drop('outcome', axis=1), df['outcome'], test_size=0.2, random_state=42)

# Train a random forest classifier on the training data
rf = RandomForestClassifier(n_estimators=100, random_state=42)
rf.fit(X_train, y_train)

# Define a function to make predictions
def predict_diabetes(age, sex, bmi, bp, s1, s2, s3, s4, s5, s6):
    input_data = pd.DataFrame({'age': [age], 'sex': [sex], 'bmi': [bmi], 'bp': [bp], 's1': [s1], 's2': [s2], 's3': [s3], 's4': [s4], 's5': [s5], 's6': [s6]})
    prediction = rf.predict(input_data)[0]
    return prediction

# Create a route for the home page
@app.route('/')
def index():
    return render_template('index.html')

# Create a route for the prediction page
@app.route('/predict', methods=['POST'])
def predict():
    age = int(request.form['age'])
    sex = int(request.form['sex'])
    bmi = float(request.form['bmi'])
    bp = float(request.form['bp'])
    s1 = int(request.form['s1'])
    s2 = int(request.form['s2'])
    s3 = int(request.form['s3'])
    s4 = int(request.form['s4'])
    s5 = int(request.form['s5'])
    s6 = int(request.form['s6'])
    prediction = predict_diabetes(age, sex, bmi, bp, s1, s2, s3, s4, s5, s6)
    return render_template('result.html', prediction=prediction)

if __name__ == '__main__':
    app.run(debug=True)