import os
import sys
import json
from dotenv import load_dotenv

# Load .env file
load_dotenv()

def check_kaggle_setup():
    # Check for environment variables (KAGGLE_USERNAME and KAGGLE_KEY) first
    kaggle_username = os.environ.get('KAGGLE_USERNAME')
    kaggle_key = os.environ.get('KAGGLE_KEY')
    
    if kaggle_username and kaggle_key:
        try:
            from kaggle.api.kaggle_api_extended import KaggleApi
            api = KaggleApi()
            api.authenticate()
            print("Kaggle API authenticated successfully using environment variables from .env file.")
            return
        except Exception as e:
            print(f"Error with environment variables: {str(e)}")
            print("Ensure KAGGLE_USERNAME and KAGGLE_KEY are correctly set in your .env file.")
            print("Falling back to check for kaggle.json file...")

    # Check for kaggle.json file as a fallback
    kaggle_dir = os.environ.get('KAGGLE_CONFIG_DIR', os.path.expanduser('~/.kaggle'))
    kaggle_json_path = os.path.join(kaggle_dir, 'kaggle.json')

    if not os.path.exists(kaggle_json_path):
        print(f"Error: kaggle.json not found in {kaggle_json_path}")
        print("Since you prefer using environment variables, ensure your .env file contains:")
        print("  KAGGLE_USERNAME=your_kaggle_username")
        print("  KAGGLE_KEY=your_kaggle_api_key")
        print("To get these credentials:")
        print("1. Go to https://www.kaggle.com/")
        print("2. Sign in to your Kaggle account (or create one if you don't have an account).")
        print("3. Click on your profile picture in the top-right corner and select 'Account'.")
        print("4. Scroll down to the 'API' section and click 'Create New API Token'.")
        print("5. This will download a 'kaggle.json' file containing your username and key.")
        print("6. Open kaggle.json in a text editor and copy the 'username' and 'key' values.")
        print("7. Add them to your .env file in the project root (e.g., G:\\Ezitech Internship\\Task2):")
        print("   KAGGLE_USERNAME=your_username")
        print("   KAGGLE_KEY=your_key")
        print("Alternatively, set environment variables manually:")
        print("   - Open Command Prompt and run:")
        print("     set KAGGLE_USERNAME=your_kaggle_username")
        print("     set KAGGLE_KEY=your_kaggle_api_key")
        print("   - To make them permanent, run:")
        print("     setx KAGGLE_USERNAME your_kaggle_username")
        print("     setx KAGGLE_KEY your_kaggle_api_key")
        print("   - Or add them to User Environment Variables in Windows:")
        print("     1. Search for 'Environment Variables' in Windows.")
        print("     2. Under 'User variables', add KAGGLE_USERNAME and KAGGLE_KEY.")
        print("See detailed setup instructions at https://github.com/Kaggle/kaggle-api/")
        sys.exit(1)
    else:
        try:
            with open(kaggle_json_path, 'r') as f:
                json.load(f)  # Validate JSON
            # Set secure permissions
            os.system(f'icacls "{kaggle_json_path}" /inheritance:r')
            os.system(f'icacls "{kaggle_json_path}" /grant:r "%username%:F"')
            from kaggle.api.kaggle_api_extended import KaggleApi
            api = KaggleApi()
            api.authenticate()
            print(f"kaggle.json found at {kaggle_json_path} and API authentication successful.")
        except json.JSONDecodeError:
            print(f"Error: {kaggle_json_path} is corrupted or invalid.")
            print("Please download a new kaggle.json file from https://www.kaggle.com/ or use environment variables.")
            sys.exit(1)
        except Exception as e:
            print(f"Error with Kaggle setup: {str(e)}")
            print("Please ensure your kaggle.json contains valid API credentials or use KAGGLE_USERNAME and KAGGLE_KEY in your .env file.")
            sys.exit(1)

if __name__ == '__main__':
    check_kaggle_setup()