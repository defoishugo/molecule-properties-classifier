from flask import Flask, Response, request
from servier.dataset import build_prediction_dataset

model_used = None
app = Flask("servier")


@app.route("/predict")
def predict():
    global model_used
    smiles = request.args.get("smiles")
    X = build_prediction_dataset(smiles, 2, 2048)
    model_used.X = X
    if smiles:
        prob = model_used.predict()
        return Response('{"message":' + str(prob) + '}',
                        status=200, mimetype='application/json')
    else:
        return Response('{"message":"KO"}', status=400,
                        mimetype='application/json')


def serve(model):
    global model_used
    model_used = model
    app.run(host='0.0.0.0', port='8000')
    return 0
