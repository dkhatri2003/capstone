import config
import preprocess_prod as pp
import pandas as pd
import json

from flask import Flask, request, Response, render_template
from flask_cors import cross_origin

app = Flask(__name__)

@app.route('/')
def home():
    return render_template('index.html')

@app.route('/v1/accident_alert',methods=['POST'])
def accident_alert():
    
    data=request.get_json(force=True) 

    cols=list(data.keys())
    values=[list(data.values())]
    data=pd.DataFrame(values, columns=cols)
    
    # Prediction & Response
    prediction=pp.predict(data)[0]
    response={ "Potential Accident Level" : config.TARGET_LEVEL[prediction], "Potential Accident Description" : config.TARGET_LEVEL_DESC[prediction]}
    
    return Response(json.dumps(response),  mimetype='application/json')

# Chatbot API
@app.route('/v2/accident_alert', methods=['POST'])
@cross_origin()
def accident_alert_v2():   
    req = request.get_json(silent=True, force=True)
    result = req.get("queryResult")
    data = result.get("parameters")
    
    cols=list(data.keys())
    values=[list(data.values())]
    data=pd.DataFrame(values, columns=cols)
 
    # Prediction & Response
    prediction=pp.predict(data)[0]    
    response = 'Potential Accident Level is '+ config.TARGET_LEVEL[prediction] + ' & ' + 'Potential Accident Description: ' + config.TARGET_LEVEL_DESC[prediction]
    
    response={"fulfillmentText": response}
    return Response(json.dumps(response),  mimetype='application/json')
    
if __name__ == "__main__":
    app.run(debug=True)
