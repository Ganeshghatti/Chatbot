from flask import Blueprint, request, jsonify
from crewai import Agent, Task, Crew, LLM
from crewai_tools import PDFSearchTool
from dotenv import load_dotenv
import os

# Load environment variables
load_dotenv()
api_key = os.getenv("OPENAI_API_KEY")
llm = LLM(model="groq/gemma2-9b-it", temperature=0.7, max_tokens=1500, api_key=api_key)

# Initialize PDF tool
pdf_tool = PDFSearchTool(
    pdf='thesquirrel.pdf',
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

chat_bp = Blueprint('chat', __name__)

@chat_bp.route('/chat/squirrel', methods=['POST'])
def handle_chat():
    try:
        data = request.json
        if not data or not isinstance(data, list):
            return jsonify({'error': 'Invalid chat format'}), 400

        # Extract the actual message and recent chat history
        user_message = data[-1].get('message', '') if data else ''
        recent_messages = data[-5:] if len(data) > 5 else data
        print(f"User message: {user_message}")

        # Create research agent
        research_agent = Agent(
            role='PDF Research Specialist',
            goal='Extract relevant information from PDF based on chat history',
            backstory="""You are an expert at finding and explaining information from The Squirrel's documentation.
            You provide clear, accurate answers based on the PDF content. Focus on finding specific details about:
            - Product features and capabilities
            - Pricing information
            - Technical specifications
            - Integration details
            - Company policies""",
            tools=[pdf_tool],
            verbose=True,
            llm=llm
        )

        # Create customer support agent
        support_agent = Agent(
            role='Customer Support Specialist',
            goal='Format and personalize responses based on chat context',
            backstory="""You are a friendly customer support specialist for The Squirrel who excels at:
            - Formatting technical information into easy-to-read responses within 250 characters
            - Using appropriate formatting (bullet points, new lines) for clarity
            - Maintaining context from previous messages
            - Providing concise, relevant answers
            - If information is missing or unclear, use:
              "I apologize, but I am not sure about this. 
              For the most accurate information, please contact our team at info@thesquirrel.site"
            """,
            verbose=True,
            llm=llm
        )

        # Create research task
        research_task = Task(
            description=f"""
            Search the PDF for information about: {user_message}
            
            Example format for queries:
            - "What are the key features of [product]?"
            - "How does [feature] work?"
            - "What are the pricing options?"
            
            Be specific and thorough in your search.
            """,
            expected_output="Detailed information from the PDF content",
            agent=research_agent
        )

        # Create support task
        support_task = Task(
            description=f"""
            Format and personalize this technical information for the customer.
            
            Recent chat history: {recent_messages}
            Research results: {{research_task_result}}
            
            Guidelines:
            1. Use appropriate formatting:
               - Bullet points for lists
               - New lines (\n) for readability
               - Bold for important points
            2. Keep the response concise and relevant
            3. Reference previous context when appropriate
            4. If information is missing or unclear, use:
               "I apologize, but I don't have this specific information in my database. 
               For the most accurate information, please contact our team at info@thesquirrel.site"
            """,
            expected_output="A well-formatted, contextual response",
            agent=support_agent
        )

        # Create and run crew
        crew = Crew(
            agents=[research_agent, support_agent],
            tasks=[research_task, support_task],
            verbose=True
        )
        
        result = crew.kickoff()
        
        # Format the response
        return jsonify({
            'response': str(result).strip(),
            'status': 'success'
        })

    except Exception as e:
        print(f"Error: {str(e)}")
        return jsonify({
            'response': "I apologize, but I don't have this specific information in my database. For the most accurate information, please contact our team at info@thesquirrel.site",
            'status': 'error'
        }), 500