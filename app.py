from flask import Flask, request, jsonify
from crewai import Agent, Task, Crew
from crewai_tools import PDFSearchTool
from dotenv import load_dotenv
import os

# Load environment variables
load_dotenv()
api_key = os.getenv("OPENAI_API_KEY")
llm = LLM(model="groq/gemma2-9b-it", temperature=0.7, max_tokens=1500, api_key=api_key)

app = Flask(__name__)

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

# Create Customer Support Agent
support_agent = Agent(
    role='Customer Support Specialist',
    goal='Provide accurate and helpful responses to customer queries using company documentation',
    backstory="""You are an experienced customer support specialist at The Squirrel.
    Your goal is to provide clear, concise, and accurate responses based on the company documentation.
    If information is not available in the documents, reply with contact information""",
    tools=[pdf_tool],
    verbose=True,
    llm=llm
)

def get_agent_response(query):
    task = Task(
        description=f"""Answer the following customer query: {query}
        1. Search the documentation using the PDF tool
        2. Provide a clear and concise response
        3. If information is not found, respond with "I apologize, but I don't have that information. Please contact us at info@thesquirrel.site"
        4. Keep responses professional and to the point""",
        expected_output="A clear, professional response to the customer's query based on the company documentation.",
        agent=support_agent
    )
    
    crew = Crew(
        agents=[support_agent],
        tasks=[task],
        verbose=True
    )
    
    result = crew.kickoff()
    # Extract the string response from CrewOutput
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

if __name__ == '__main__':
    app.run(host='0.0.0.0', port=8001) 