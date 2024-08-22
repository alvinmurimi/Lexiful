from flask import Flask, render_template, request, jsonify
from main import Lexiful

app = Flask(__name__)

# Load the model
matcher = Lexiful('config.yaml')

@app.route('/')
def index():
    return render_template('index.html')

@app.route('/suggest', methods=['POST'])
def suggest():
    input_text = request.json['input']
    suggestions = matcher.match(input_text, max_matches=10)
    return jsonify(suggestions)

if __name__ == '__main__':
    app.run(debug=True)
