from flask import Flask, request, jsonify


app = Flask(__name__)

@app.route('/hello')
def hello():
    return "Hello"

if __name__=="__main__":
    app.run(debug=True, host='127.0.0.1')
