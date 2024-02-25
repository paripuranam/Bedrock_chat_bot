import boto3
import json
from flask import Flask, request, jsonify
from flask_cors import CORS  
import os
from dotenv import load_dotenv

load_dotenv()  

AWS_ACCESS_KEY_ID = os.getenv('AWS_ACCESS_KEY_ID')
AWS_SECRET_ACCESS_KEY = os.getenv('AWS_SECRET_ACCESS_KEY')
AWS_DEFAULT_REGION = os.getenv('AWS_DEFAULT_REGION')

app = Flask(__name__)
CORS(app)  

bedrock_runtime = boto3.client('bedrock-runtime', region_name='us-east-1')

@app.route('/generate-text', methods=['POST'])
def generate_text():
    prompt = request.json.get('prompt')
    kwargs = {
        "modelId": "ai21.j2-ultra-v1",
        "contentType": "application/json",
        "accept": "*/*",
        "body": json.dumps({
            "prompt": prompt,
            "maxTokens": 500,
            "temperature": 0,
            "topP": 1,
            "stopSequences": [],
            "countPenalty": {"scale": 0},
            "presencePenalty": {"scale": 0},
            "frequencyPenalty": {"scale": 0}
        })
    }

    response = bedrock_runtime.invoke_model(**kwargs)
    response_body = json.loads(response.get("body").read())
    completion = response_body.get('completions')[0].get('data').get('text')
    
    return jsonify({'completion': completion})

if __name__ == '__main__':
    app.run(debug=True)