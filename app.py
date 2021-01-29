import config
import preprocessing as pp

import numpy as np
import pandas as pd
import json

from flask import Flask, request, jsonify, make_response, Response


app = Flask(__name__)
@app.route('/v1/accident_alert',methods=['POST'])
def accident_alert():
    
    data=request.get_json(force=True) 

    cols=list(data.keys())
    values=[list(data.values())]
    data=pd.DataFrame(values, columns=cols)
    
    # Prediction & Response
    #prediction=pp.predict(data)[0]
    #response={ "Potential Accident Level" : config.TARGET_LEVEL[prediction], 
                         "Potential Accident Description" : config.TARGET_LEVEL_DESC[prediction]}
    
    #return Response(json.dumps(response),  mimetype='application/json')
    return 'Hello World!'

if __name__ == "__main__":
    app.run(host='localhost', port=5000) 
