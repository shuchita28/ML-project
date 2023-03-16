import pickle
from Flask import Flask, request, render_template
import numpy as np
import pandas as pd
from sklearn.preprocessing import StandardScaler
from src.pipeline.predict_pipeline import CustomData, PredictPipeline

application = Flask(__name__)
app = application

## Route for home page

@app.route('/')
def index():
    return render_template('index.html')

@app.route('/predictdata', methods=['GET', 'POST'])
def predict_datapoint():
    if request.method == 'GET':
        return render_template('home.html')
    else:
        data = CustomData(
            B12 = request.form.get('B12'),
            Age = request.form.get('Age'),
            Gender = float(request.form.get('Gender')),
            DPQ010 = request.form.get('DPQ010'),
            DPQ020 = request.form.get('DPQ020'),
            DPQ030 = request.form.get('DPQ030'),
            DPQ040 = request.form.get('DPQ040'),
            DPQ050 = request.form.get('DPQ050'),
            DPQ060 = request.form.get('DPQ060'),
            DPQ070 = request.form.get('DPQ070'),
            DPQ080 = request.form.get('DPQ080'),
            DPQ090 = request.form.get('DPQ090'),
            SLQ050 = request.form.get('SLQ050'),
            SLQ060 = request.form.get('SLQ060'),
            IND235 = request.form.get('IND235')
        )
        pred_df = data.get_data_as_data_frame()
        print(pred_df)

        predict_pipeline = PredictPipeline()
        results = predict_pipeline.predict(pred_df)
        return render_template('home.html', results = results[0])
    
if __name__=="__main__":
    app.run(host = "0.0.0.0", debug=True)