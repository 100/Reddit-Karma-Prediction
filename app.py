from flask import Flask, render_template, request, jsonify, make_response, \
    flash, abort
from wtforms import Form, TextAreaField, validators
from flask_limiter import Limiter
from createFullClassifier import vectorize
import numpy
import json, os
try:
   import cPickle as pickle
except:
   import pickle

app = Flask(__name__)
app.secret_key = os.environ.get("FLASK_SECRET_KEY", "star-this-repo-pls")
limiter = Limiter(app)

class CommentForm(Form):
    comment = TextAreaField('comment',
        [validators.required(), validators.Length(min = 2, max = 1000)])


def validateRequest(type, text):
    if type not in ['binary', 'bins']:
        return False
    if not isinstance(text, unicode) or len(str(text)) < 2 or len(str(text)) > 1000:
        return False
    return True

def classifyComment(type, text):
    if type == 'bins':
        with open('pickles/fullClassifier.pkl', 'rb') as pickleFile:
            fullClf = pickle.load(pickleFile)
        with open('pickles/ngramBinaryClf.pkl', 'rb') as pickleFile:
            ngram = pickle.load(pickleFile)
        with open('pickles/blobber.pkl', 'rb') as pickleFile:
            blobber = pickle.load(pickleFile)
        with open('pickles/swearList.pkl', 'rb') as pickleFile:
            swears = pickle.load(pickleFile)
        vector = vectorize(blobber, text, ngram)
        prediction = fullClf.predict([vector])[0]
        return vector, prediction
    if type == 'binary':
        with open('pickles/ngramBinaryClf.pkl', 'rb') as pickleFile:
            ngram = pickle.load(pickleFile)
        prediction = ngram.predict([text])[0]
        return prediction


@app.route('/')
def index():
    return render_template('index.html')

@app.route('/try', methods=['GET', 'POST'])
def tryIt():
    form = CommentForm(request.form)
    if request.method == 'POST':
        if form.validate():
            comment = form.comment.data
            vector, prediction = classifyComment('bins', comment)
            if vector[9] == 1:
                vector[9] == 'positive'
            else:
                vector[9] == 'negative'
            return render_template('try.html', form = form, vector = vector,
                classification = prediction)
        else:
            flash('Error! Hint: Comment must be between 2 and 1000 characters.')
    return render_template('try.html', form = form)

@app.route('/visualize')
def visualize():
    with open('pickles/elements.pkl', 'rb') as pickleFile:
        elements = pickle.load(pickleFile)
    return render_template('visualize.html', elements = json.dumps(elements))

@app.route('/apidocs')
def apidocs():
    return render_template('apidocs.html')


@app.route('/api')
@limiter.limit("10 per minute")
def api():
    if request.args is None:
        abort(400)
    if request.args.get('type') is not None:
        clfType = request.args.get('type')
    else:
        abort(400)
    if request.args.get('text') is not None:
        text = request.args.get('text')
    else:
        abort(400)
    if validateRequest(clfType, text) == False:
        abort(400)
    text = str(text)
    if clfType == 'bins':
        __, prediction = classifyComment('bins', text)
    else:
        prediction = classifyComment('binary', comment)
    status = {'status': 'ok', 'type': clfType, 'prediction': prediction}
    return jsonify(status)

@app.errorhandler(400)
def incorrect(error):
    return make_response(jsonify({'status': 'error',
        'error': 'Incorrect or inappropriate parameters provided.'}), 400)

@app.errorhandler(429)
def ratelimitExceeded(error):
    return make_response(jsonify({'status': 'error',
        'error': 'Rate limit exceeded.'}), 429)


if __name__ == '__main__':
    port = int(os.environ.get('PORT', 5000))
    app.run(debug=True, host='0.0.0.0', port=port)
