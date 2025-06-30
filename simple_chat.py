"""
Simple example script demonstrating basic Gemini usage with LangChain.
"""

from langchain_gemini import GeminiLangChainBot
import os
from dotenv import load_dotenv

# Load environment variables from .env file
load_dotenv()

def main():
    """
    Simple demonstration of Gemini with LangChain.
    """
    
    # Check if API key is available
    api_key = os.getenv("GOOGLE_API_KEY")
    if not api_key:
        print("Please set your GOOGLE_API_KEY in the .env file or as an environment variable.")
        print("You can get your API key from: https://makersuite.google.com/app/apikey")
        return
    
    try:
        # Initialize the bot
        bot = GeminiLangChainBot()
        
        print("Gemini LangChain Bot initialized successfully!")
        print("Type 'quit' to exit, 'clear' to clear memory\n")
        
        while True:
            user_input = input("You: ").strip()
            
            if user_input.lower() == 'quit':
                print("Goodbye!")
                break
            elif user_input.lower() == 'clear':
                bot.clear_memory()
                print("Memory cleared!")
                continue
            elif not user_input:
                continue
            
            # Get response with conversation memory
            print("AI: ", end="", flush=True)
            response = bot.setup_document_qa(user_input, )
            if isinstance(response, str):
                print(response)
            elif hasattr(response, 'content'):
                print(response.content)
            else:
                print("Unexpected response format.")
            print()  # New line after streaming response
            
    except Exception as e:
        print(f"Error: {e}")

if __name__ == "__main__":
    main()
