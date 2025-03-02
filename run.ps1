# Start the Python diabetes application
Start-Process -NoNewWindow -FilePath python -ArgumentList "diabetesapp.py"

# Wait for a moment to allow the server to start
Start-Sleep -Seconds 5

# Open the default web browser to the specified URL
Start-Process "http://127.0.0.1:5000"