import subprocess
import sys
import os

def run_streamlit():
    """Run the Streamlit app"""
    current_dir = os.path.dirname(__file__)
    app_path = os.path.join(current_dir, "streamlit_app.py")
    
    print("Starting Agentic RAG Streamlit App...")
    print(f"App location: {app_path}")
    print(f"Current directory: {current_dir}")
    print("The app will open in your default browser.")
    print("Press Ctrl+C to stop the server.")
    print("\nIf the page keeps loading, check the terminal for error messages.")
    print("The app may take a moment to initialize all models.\n")
    
    # Check if the app file exists
    if not os.path.exists(app_path):
        print(f"Error: streamlit_app.py not found at {app_path}")
        return
    
    try:
        subprocess.run([
            sys.executable, "-m", "streamlit", "run", app_path,
            "--server.address", "0.0.0.0",
            "--server.port", "8501",
            "--browser.gatherUsageStats", "false",
            "--server.headless", "false",
            "--logger.level", "info"
        ])
    except KeyboardInterrupt:
        print("\nStreamlit app stopped.")
    except Exception as e:
        print(f"Error running Streamlit app: {e}")
        print("Try running: streamlit run streamlit_app.py --logger.level debug")

if __name__ == "__main__":
    run_streamlit()
