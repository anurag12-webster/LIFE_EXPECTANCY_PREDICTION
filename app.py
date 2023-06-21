from flask import Flask ,render_template, request
import pickle
import numpy as np

app = Flask(__name__)

@app.route('/')
def home():
    return render_template('index.html')

#prediction function
def ValuePredictor(to_predict_list):
    to_predict = to_predict_list[2:]
    to_predict = np.array(to_predict).reshape(1, 18)
    loaded_model = pickle.load(open("model.pkl", "rb"))
    result = loaded_model.predict(to_predict)
    return result[0]

@app.route('/result', methods = ['POST'])
def result():
    if request.method == 'POST':
        to_predict_list = request.form.to_dict()
        to_predict_list = list(to_predict_list.values())
        #to_predict_list = list(map(int, to_predict_list))
        #print(to_predict_list)
        result = ValuePredictor(to_predict_list)
        return render_template("index.html", prediction = 'Life Expectancy(in years): {}'.format(result))


if __name__ == "__main__":
    app.run()