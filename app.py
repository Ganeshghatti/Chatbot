from flask import Flask, jsonify
from flask_cors import CORS
from squirrel_bot import chat_bp
from socialhardware_bot import social_bp

app = Flask(__name__)
CORS(app)

# Register both blueprints
app.register_blueprint(chat_bp)
app.register_blueprint(social_bp)

@app.route('/')
def index():
    return jsonify({'message': 'Welcome to the Chatbot API'})

if __name__ == '__main__':
    app.run(host='0.0.0.0', port=8002) 