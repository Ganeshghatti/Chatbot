from flask import Flask, request, jsonify
from flask_cors import CORS
from crewai import Agent, Task, Crew, LLM
from crewai_tools import PDFSearchTool
from dotenv import load_dotenv
import os

# Load environment variables
load_dotenv()
api_key = os.getenv("OPENAI_API_KEY")
llm = LLM(model="groq/gemma2-9b-it", temperature=0.7, max_tokens=1500, api_key=api_key)

app = Flask(__name__)
CORS(app)  # Enable CORS for all routes

# Initialize PDFSearchTool
pdf_tool = PDFSearchTool(
    pdf='companydata.pdf',
    config=dict(
        llm=dict(
            provider="groq",
            config=dict(
                model="mixtral-8x7b-32768",
                temperature=0.7,
            ),
        ),
        embedder=dict(
            provider="google",
            config=dict(
                model="models/embedding-001",
            ),
        ),
    )
)

# Improved Customer Support Agent
support_agent = Agent(
    role='Customer Support Representative',
    goal='Provide friendly and helpful customer support while maintaining professionalism',
    backstory="""You are Sarah, a friendly customer support representative at The Squirrel. 
    You have extensive knowledge of the company's services and products.
    You always:
    - Greet customers warmly
    - Show empathy and understanding
    - Provide clear, accurate information
    - Use a conversational, friendly tone
    - End conversations professionally
    - If information isn't in the documentation, provide contact details politely
    
    Your responses should feel natural and helpful, not robotic.""",
    tools=[pdf_tool],
    verbose=True,
    llm=llm
)

def get_agent_response(query):
    task = Task(
        description=f"""Handle this customer inquiry: {query}

        Guidelines:
        1. Start with a warm greeting
        2. Show understanding of the customer's query
        3. Search the documentation thoroughly
        4. Provide clear, helpful information in a conversational tone
        5. If information isn't available, say something like:
           "I apologize, but I don't have that specific information in my database. 
           For the most accurate information, please contact our team at info@thesquirrel.site 
           or visit our website at thesquirrel.site"
        6. End with a polite closing and offer further assistance
        
        Remember to maintain a friendly, helpful tone throughout the conversation.""",
        expected_output="A friendly, helpful, and professional response that addresses the customer's needs.",
        agent=support_agent
    )
    
    crew = Crew(
        agents=[support_agent],
        tasks=[task],
        verbose=True
    )
    
    result = crew.kickoff()
    return str(result)

@app.route('/chat', methods=['POST'])
def chat():
    try:
        data = request.json
        if not data or not isinstance(data, list) or len(data) == 0:
            return jsonify({'error': 'Invalid chat format'}), 400

        # Get the last user message
        last_message = data[-1]
        if last_message.get('role') != 'user' or 'message' not in last_message:
            return jsonify({'error': 'Invalid message format'}), 400

        response = get_agent_response(last_message['message'])
        return jsonify({
            'response': response,
            'status': 'success'
        })

    except Exception as e:
        return jsonify({'error': str(e), 'status': 'error'}), 500

@app.route('/')
def index():
    return jsonify({'message': 'Welcome to the Squirrel Chatbot API'})

if __name__ == '__main__':
    app.run(host='0.0.0.0', port=8002) 