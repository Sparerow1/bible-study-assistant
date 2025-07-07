import os
import sys

class Validation:
    """Class for validating user inputs and configurations."""
    @staticmethod
    def get_resource_path(relative_path):
        """Get absolute path to resource, works for dev and for PyInstaller"""
        try:
            # PyInstaller creates a temp folder and stores path in _MEIPASS
            base_path = sys._MEIPASS 
            print(f"Running from PyInstaller temp dir: {base_path}")
        except Exception:
            base_path = os.path.abspath(".")
            print(f"Running from source: {base_path}")
        
        full_path = os.path.join(base_path, relative_path)
        print(f"Looking for data file at: {full_path}")
        return full_path
    
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