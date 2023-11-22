from flask import Flask, render_template, request
import pickle
from sklearn.preprocessing import MinMaxScaler

app = Flask(__name__)

with open('gb_model.pkl', 'rb') as model_file:
    loaded_gb_model = pickle.load(model_file)

# Routes
@app.route('/')
def index():
    return render_template('index.html')

@app.route('/predict', methods=['POST'])
def predict():
    if request.method == 'POST':

        user_id = request.form['user_id']
        age = float(request.form['age'])
        gender = 1 if request.form['gender'] == 'Male' else 0
        salary = float(request.form['salary'])
        
        prediction = loaded_gb_model.predict([[age, gender, salary]])

        if prediction[0] == 1:
            result_message = f"Customer with ID {user_id} is likely to purchase a car."
        else:
            result_message = f"Customer with ID {user_id} is unlikely to purchase a car."
        
        return render_template('result.html', prediction=prediction[0], message=result_message)

if __name__ == '__main__':
    app.run(debug=True)
