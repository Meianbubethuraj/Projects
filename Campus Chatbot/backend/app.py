from flask import Flask, request, jsonify # type: ignore
from database import get_answer
import openai # type: ignore

app = Flask(__name__)

# OpenAI API Key (replace with your key)
openai.api_key = " "

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

