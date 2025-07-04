import os
import sys
import traceback
from qa.qa_class import BibleQASystem


def main():
    """Main entry point."""
    try:
        bible_qa = BibleQASystem()
        
        if bible_qa.initialize():
            bible_qa.run_interactive_session()
            return 0
        else:
            print("❌ Failed to initialize Bible Q&A system")
            return 1
            
    except Exception as e:
        print(f"❌ Critical error: {e}")
        if os.getenv("DEBUG", "").lower() == "true":
            traceback.print_exc()
        return 1


if __name__ == "__main__":
    try:
        exit_code = main()
        sys.exit(exit_code)
    except Exception as e:
        print(f"❌ Fatal error: {e}")
        sys.exit(1)