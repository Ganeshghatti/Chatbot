from flask import Blueprint, request, jsonify
from crewai import Agent, Task, Crew, LLM
from crewai_tools import PDFSearchTool
from dotenv import load_dotenv
import os

# Load environment variables
load_dotenv()
api_key = os.getenv("OPENAI_API_KEY")
llm = LLM(model="groq/gemma2-9b-it", temperature=0.7, max_tokens=1500, api_key=api_key)

# Initialize PDF tool for SocialHardware
pdf_tool = PDFSearchTool(
    pdf='socialhardware.pdf',
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

social_bp = Blueprint('social', __name__)

@social_bp.route('/chat/socialhardware', methods=['POST'])
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
            backstory="""You are an expert at finding and explaining information from SocialHardware's documentation.
            You provide clear, accurate answers based on the PDF content. Focus on finding specific details about:
            - Product features and capabilities
            - Technical capabilities
            - Pricing and packages
            - Installation requirements
            - Support services""",
            tools=[pdf_tool],
            verbose=True,
            llm=llm
        )

        # Create customer support agent
        support_agent = Agent(
            role='Customer Support Specialist',
            goal='Format and personalize responses based on chat context',
            backstory="""You are a technical support specialist for SocialHardware who excels at:
            - Formatting technical information into easy-to-read responses within 250 characters
            - Using appropriate formatting for technical specifications
            - Maintaining context from previous messages
            - Providing precise technical answers
            - If information is missing or unclear, use:
              "I apologize, but I am not sure about this. 
              For the most accurate information, please contact our support team at sh.lab@socialhardware.co.in"
            """,
            verbose=True,
            llm=llm
        )

        # Create research task
        research_task = Task(
            description=f"""
            Search the PDF for information about: {user_message}
            
            Example format for queries:
            - "What are the specifications of [product]?"
            - "How does [technical specifications] of [product]?"
            - "What are the installation requirements?"
            
            Be specific and thorough in your search.
            """,
            expected_output="Detailed technical information from the PDF content",
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
               - Bullet points for technical specifications
               - New lines (\n) for readability
               - Bold for important technical details
            2. Keep responses technically precise
            3. Reference previous context when appropriate
            4. If information is missing or unclear, use:
               "I apologize, but I am not sure about this. 
               For the most accurate information, please contact our support team at sh.lab@socialhardware.co.in"
            """,
            expected_output="A well-formatted, technical response",
            agent=support_agent
        )

        # Create and run crew
        crew = Crew(
            agents=[research_agent, support_agent],
            tasks=[research_task, support_task],
            verbose=True
        )
        
        result = crew.kickoff()
        
        return jsonify({
            'response': str(result).strip(),
            'status': 'success'
        })

    except Exception as e:
        print(f"Error: {str(e)}")
        return jsonify({
            'response': "I apologize, but I don't have this specific information. For the most accurate information, please contact our support team at support@socialhardware.com",
            'status': 'error'
        }), 500 