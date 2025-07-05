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