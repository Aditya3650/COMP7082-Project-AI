from flask import Flask, request, jsonify
from flask_cors import CORS
from sentence_transformers import CrossEncoder
import torch

app = Flask(__name__)
CORS(app)  # Enable CORS

# Load CrossEncoder model with sigmoid activation
model = CrossEncoder("cross-encoder/ms-marco-MiniLM-L-6-v2", default_activation_function=torch.nn.Sigmoid())

@app.route('/similarity', methods=['POST'])
def get_similarity():
    data = request.get_json()
    sentence1 = data['sentence1']
    sentence2 = data['sentence2']

    scores = model.predict([(sentence1, sentence2)])
    similarity_percentage = scores[0] * 100

    return jsonify({
        'similarity': f"{similarity_percentage:.2f}"
    })

if __name__ == '__main__':
    app.run(debug=True)
