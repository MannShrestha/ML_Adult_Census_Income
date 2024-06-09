from flask import Flask, render_template, request, jsonify
from censusIncome.pipeline.prediction_pipeline import PredictionPipeline, CustomData

app = Flask(__name__)

@app.route("/", methods = ['GET', 'POST'])
def prediction_data():
    if request.method == "GET":
        return render_template("home.html")
    
    else:
        # Get the input data from the form in the home.html
        data = CustomData(
            age = int(request.form.get("age")),
            workclass = int(request.form.get("workclass")),
            education_num = int(request.form.get("education_num")),
            marital_status = int(request.form.get("marital_status")),
            occupation = int(request.form.get("occupation")),
            relationship = int(request.form.get("relationship")),
            race = int(request.form.get("race")),
            sex = int(request.form.get("sex")),
            capital_gain = int(request.form.get("capital_gain")),
            capital_loss = int(request.form.get("capital_loss")),
            hours_per_week = int(request.form.get("hours_per_week")),
        
        )

        pred_df = data.get_data_as_data_frame()
        print(pred_df)
        print("Before Prediction")

        prediction_pipeline = PredictionPipeline()
        print("Mid Prediction")

        results = prediction_pipeline.predict(pred_df)
        print("After prediction")

        if results == 0:
            return render_template("index.html", final_result = "Your Yearly Income is Less than Equal to 50k:{}".format(results) )

        elif results == 1:
            return render_template("index.html", final_result = "Your Yearly Income is More than 50k:{}".format(results) )
    
if __name__ == "__main__":
    app.run(host="0.0.0.0", debug = True)

