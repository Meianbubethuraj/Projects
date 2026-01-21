from flask import Flask, request, jsonify # type: ignore
from database import get_answer
import openai # type: ignore

app = Flask(__name__)

# OpenAI API Key (replace with your key)
openai.api_key = "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"

@app.route('/chat', methods=['POST'])
def chat():
    data = request.get_json()
    user_query = data.get("query")

    # Check in database
    answer = get_answer(user_query)
    if answer:
        return jsonify({"answer": answer})

    # If no answer in database, return college contact
    contact_info = "Contact with this Number: 123-456-7890 or visit college website: www.college.com"
    
    return jsonify({"answer": contact_info})

if __name__ == '__main__':
    app.run(debug=True)
