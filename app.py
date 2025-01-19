from flask import Flask, request, jsonify
from flask_cors import CORS
from crewai import Agent, Task, Crew, LLM
from crewai_tools import PDFSearchTool
from dotenv import load_dotenv
import os

# Load environment variables
load_dotenv()
api_key = os.getenv("OPENAI_API_KEY")
llm = LLM(model="groq/llama3-8b-8192", temperature=0.7, max_tokens=1500, api_key=api_key)

app = Flask(__name__)
CORS(app)

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

# Research Agent - Specialized in gathering information
research_agent = Agent(
    role='Information Research Specialist',
    goal='Extract and organize relevant information from company documentation',
    backstory="""You are an expert researcher who excels at finding specific information.
    Your role is to:
    - Search through documentation for relevant details
    - Extract and organize key information
    - Provide clear, factual responses
    - Focus only on the current query""",
    tools=[pdf_tool],
    llm=llm
)

# Response Formatter Agent - Combines formatting and cleaning
formatter_agent = Agent(
    role='Technical Response Formatter',
    goal='Format and clean information into precise bullet points',
    backstory="""You are a precise formatting system that:
    - Converts information into clean bullet points
    - Removes all greetings and pleasantries
    - Uses only technical language
    - Never adds commentary or personal pronouns
    - Formats output as "ProductName: * feature"
    - Keeps responses direct and concise""",
    tools=[],
    llm=llm
)

def get_agent_response(messages):
    latest_message = messages[-1]['message']
    
    # Research Task
    research_task = Task(
        description=f"""Search the documentation for: {latest_message}""",
        expected_output="Raw information about products and features",
        agent=research_agent
    )
    
    research_result = Crew(
        agents=[research_agent],
        tasks=[research_task],
        verbose=True
    ).kickoff()
    
    # Format and Clean Task
    format_task = Task(
        description=f"""Format this information into clean bullet points:
        INPUT: {research_result}
        
        REQUIREMENTS:
        1. Start with product name
        2. List features with asterisk bullets
        3. NO greetings or pleasantries
        4. NO commentary or personal pronouns
        5. Format exactly as:
           ProductName:
           * feature
           * feature""",
        expected_output="Clean, technical bullet points without any conversational elements",
        agent=formatter_agent
    )
    
    final_result = Crew(
        agents=[formatter_agent],
        tasks=[format_task],
        verbose=True
    ).kickoff()
    
    return str(final_result)

@app.route('/chat', methods=['POST'])
def chat():
    try:
        data = request.json
        if not data or not isinstance(data, list):
            return jsonify({'error': 'Invalid chat format'}), 400

        response = get_agent_response(data)
        return jsonify({
            'response': response,
            'status': 'success'
        })
    except Exception as e:
        return jsonify({'error': str(e), 'status': 'error'}), 500

if __name__ == '__main__':
    app.run(debug=True, host='0.0.0.0', port=8002)