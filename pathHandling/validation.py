import os
import sys

class Validation:
    """Class for validating user inputs and configurations."""
    
    @staticmethod
    def load_env_from_bundle():
        """Load .env file from PyInstaller bundle or local directory"""
        env_path = Validation.get_resource_path('.env')
        
        if os.path.exists(env_path):
            print(f"✅ Found .env file at: {env_path}")
            return env_path
        else:
            print(f"❌ .env file not found at: {env_path}")
            # List contents of the directory for debugging
            directory = os.path.dirname(env_path)
            if os.path.exists(directory):
                print(f"Contents of {directory}:")
                for item in os.listdir(directory):
                    print(f"  - {item}")
            return None