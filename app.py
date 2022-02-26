from flask import Flask, request, jsonify, render_template
import pickle
import numpy as np

# Create flask app
app = Flask(__name__,template_folder='template')

# Load the pickle model
model = pickle.load(open("model.pkl", "rb"))


@app.route("/")
def Home():
    return render_template("index (1).html")
    app.run(debug=True)


@app.route("/predict", methods=["POST"])
def predict():
    float_features = [float(x) for x in request.form.values()]
    features = [np.array(float_features)]
    prediction = model.predict(features)

    return render_template("index (1).html", prediction_text="The heart disease chances are {}".format(prediction))


if __name__ == "__main__":
    app.run(debug=True)
