from flask import Flask, render_template, request
import prediction

app = Flask(__name__)

@app.route('/')
def hello():
    return render_template('index2.html', name="")

@app.route('/', methods=['POST'])
def submit():
    tweet = request.form['tweet']
    pred = prediction.predict(tweet)
    return render_template('index2.html', name=pred)



if __name__=="__main__":   
    app.run(debug=True)