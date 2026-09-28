from pyngrok import ngrok
import subprocess
import time

NGROK_AUTH_TOKEN = "PASTE_YOUR_NGROK_TOKEN_HERE"

ngrok.set_auth_token(NGROK_AUTH_TOKEN)

streamlit_process = subprocess.Popen(["streamlit", "run", "app.py", "--server.port", "8501"])
time.sleep(4)

public_url = ngrok.connect(8501)
print("=" * 60)
print(f"YOUR PUBLIC LINK: {public_url}")
print("Share this link with anyone — it works from any device.")
print("=" * 60)

try:
    streamlit_process.wait()
except KeyboardInterrupt:
    ngrok.kill()
    streamlit_process.terminate()
