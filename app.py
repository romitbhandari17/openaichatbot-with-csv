from __future__ import print_function
from config import *
import sys
import logging
import requests
from flask import Flask, jsonify, render_template
from flask_cors import CORS, cross_origin
from flask import request

#from handle_file import handle_file
from read_qa_embeddings import answer_question_from_embeddings
from create_embeddings_rankings import create_embeddings

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
    handlers=[
        logging.FileHandler("debug.log"),
        logging.StreamHandler(sys.stdout)
    ]
)

def create_app():
    #pinecone_index = load_pinecone_index()
    #tokenizer = tiktoken.get_encoding("gpt2")
    #session_id = str(uuid.uuid4().hex)
    app = Flask(__name__)
    #app.pinecone_index = pinecone_index
    #app.tokenizer = tokenizer
    #app.session_id = session_id
    # log session id
    #logging.info(f"session_id: {session_id}")
    #app.config["file_text_dict"] = {}
    CORS(app, supports_credentials=True)

    return app

app = create_app()

@app.route(f"/create_embeddings", methods=["GET"])
@cross_origin(supports_credentials=True)
def process_file():
    try:
        #file = request.files['file']
        #logging.info(str(file))
        #handle_file(
        #    file, app.session_id, app.pinecone_index, app.tokenizer)
        '''with open('create_embeddings_rankings.py', 'r') as file:
            code = file.read()
        
        exec(code)'''

        create_embeddings_response = create_embeddings()
        return create_embeddings_response
    except Exception as e:
        logging.error(str(e))
        return jsonify({"success": False})


@app.route(f"/process_create_embeddings", methods=["GET"])
def execute_process_file():
    try:
        #file_path = "data/fia_2022_formula_1_sporting_regulations_-_issue_9_-_2022-10-19_0.pdf"  # Replace with the actual file path
        file_path = "data/wur_ranking_summary.csv"

        url = "http://127.0.0.1:8080/create_embeddings"  # Replace with the actual URL of your server

        files = {'file': open(file_path, 'rb')}

        response = requests.post(url, files=files)

        if response.status_code == 200:
            return "File processing successful"
        else:
            return "File processing failed"
    except Exception as e:
        logging.error(str(e))
        return jsonify({"success": False})
    

@app.route(f"/answer_question", methods=["POST"])
@cross_origin(supports_credentials=True)
def answer_question():
    try:
        params = request.get_json()
        print(params)
        question = params["question"]

        answer_question_response = answer_question_from_embeddings(
            question)
        return answer_question_response
    except Exception as e:
        return str(e)
    

@app.route("/healthcheck", methods=["GET"])
@cross_origin(supports_credentials=True)
def healthcheck():
    return "OK"

@app.route('/chatbot', methods=['GET'])
def chatbot_ui():
    return render_template('chatbot.html')

if __name__ == "__main__":
    app.run(debug=True, port=SERVER_PORT, threaded=True)