from flask import Flask, request, jsonify
from flask_cors import CORS
from sentence_transformers import CrossEncoder
import torch
from groq import Groq
import os
from dotenv import load_dotenv

# Load environment variables from .env file
load_dotenv()

app = Flask(__name__)
CORS(app, resources={r"/*": {"origins": "*"}})

# Initialize the sentence similarity model
similarity_model = CrossEncoder("cross-encoder/ms-marco-MiniLM-L-6-v2", default_activation_function=torch.nn.Sigmoid())

# Get the Groq API key from environment variables
groq_api_key = os.getenv("GROQ_API_KEY")

# Initialize Groq client with the API key
client = Groq(api_key=groq_api_key)

# Define your routes (for example, the /ai_remarks route)
@app.route('/ai_remarks', methods=['POST'])
def get_ai_remarks():
    data = request.get_json()
    sentence1 = data.get('sentence1')
    sentence2 = data.get('sentence2')

    if not sentence1 or not sentence2:
        return jsonify({'error': 'Both sentences are required'}), 400

    try:
        # Calculate similarity score
        similarity_score = similarity_model.predict([(sentence1, sentence2)])[0] * 100

        # Generate AI remarks
        chat_completion = client.chat.completions.create(
            messages=[
                {"role": "system", "content": "You are an AI assistant providing feedback."},
                {"role": "user", "content": f"Provide remarks on the comparison of these two sentences: '{sentence1}' and '{sentence2}'. The similarity score between them is {similarity_score:.2f}%. Provide feedback as if grading a student's answer, with sentence 1 as the answer key and sentence 2 as the student's response."}
            ],
            model="llama3-8b-8192"
        )

        ai_remarks = chat_completion.choices[0].message.content

        return jsonify({
            'similarity_score': f"{similarity_score:.2f}",
            'ai_remarks': ai_remarks
        })

    except Exception as e:
        print(f"Error during Groq API call: {str(e)}")
        return jsonify({'error': 'Something went wrong with the Groq API request'}), 500

if __name__ == '__main__':
    app.run(debug=True)
