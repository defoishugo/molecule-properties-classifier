from flask import Flask, Response, request
from servier.model import import_model, predict_model

model = None
app = Flask("servier")


@app.route("/predict")
def predict():
    global model
    smiles = request.args.get("smiles")
    if smiles:
        prob = predict_model(model, smiles, 2, 2048)
        return Response('{"message":' + str(prob) + '}',
                        status=200, mimetype='application/json')
    else:
        return Response('{"message":"KO"}', status=400,
                        mimetype='application/json')


def serve():
    global model
    model = import_model()
    app.run(host='0.0.0.0', port='8000')
    return 0
