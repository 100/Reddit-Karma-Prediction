from flask import Flask, render_template, request, jsonify, make_response
from flask_limiter import Limiter
try:
   import cPickle as pickle
except:
   import pickle

app = Flask(__name__)
limiter = Limiter(app)

def validateRequest(type, text):
    if type not in ['binary', 'bins']:
        return False
    if type(text) not in [unicode, str] or len(text) <= 0 or len(text) > 1000:
        return False
    return True


def featureWeights(binClf, comment):


@app.route('/')
def index():
    pass

@app.route('/api'):
    if request.args is None:
        abort(400)
    if request.args.get('type') not None:
        clfType = request.args.get('type') #binary or bins
    else:
        abort(400)
    if request.args.get('text') not None:
        text = request.args.get('text')
    else:
        abort(400)
    if validateRequest(clfType, text) == False:
        abort(400)

    ####
    # CLASSIFY TEXT BASED ON SELECTED CLASSIFIER
    ####

    ####
    # RETURN JSON RESPONSE WITH STATUS, KARMA PREDICTION RANGE
    ####

@app.errorhandler(400)
def incorrect(error):
    return make_response(jsonify({'status': 'Error',
        'error': 'Incorrect or inappropriate parameters provided.'}), 400)

if __name__ == '__main__':
    app.run()
