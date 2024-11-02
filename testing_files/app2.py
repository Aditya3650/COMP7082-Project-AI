from flask import Flask, request, jsonify
from flask_cors import CORS
import os
from groq import Groq

app = Flask(__name__)
CORS(app, resources={r"/*": {"origins": "*"}})

# Set your Groq API key
groq_api_key = "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"

# Initialize Groq client
client = Groq(api_key=groq_api_key)

@app.route('/ai_remarks', methods=['POST'])
def get_ai_remarks():
    data = request.get_json()
    sentence1 = data.get('sentence1')
    sentence2 = data.get('sentence2')

    if not sentence1 or not sentence2:
        return jsonify({'error': 'Both sentences are required'}), 400

    try:
        # Generate an AI response using Groq's chat completion API
        chat_completion = client.chat.completions.create(
            messages=[
                {"role": "system", "content": "You are an AI assistant providing feedback."},
                {"role": "user", "content": f"Provide remarks on the comparison of these two sentences: '{sentence1}' and '{sentence2}'. Make the remarks as if you are grading answers, and sentence 1 is the answer key, and sentence 2 is the student's answer."}
            ],
            model="llama3-8b-8192"  # Using the Groq model
        )

        # Extract the AI response text correctly from the object
        ai_text = chat_completion.choices[0].message.content

        return jsonify({'ai_response': ai_text})

    except Exception as e:
        print(f"Error during Groq API call: {str(e)}")  # Log the error
        return jsonify({'error': 'Something went wrong with the Groq API request'}), 500

if __name__ == '__main__':
    app.run(debug=True)
