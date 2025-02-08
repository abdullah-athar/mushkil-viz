#!/bin/bash

# Load environment variables if they exist
if [ -f src/mushkil_viz/server/.env ]; then
    export $(cat src/mushkil_viz/server/.env | grep -v '^#' | xargs)
fi

# Start the backend server
echo "Starting backend server..."
uvicorn mushkil_viz.server.main:app --reload --port ${BACKEND_PORT:-8001} &

# Wait a bit for the backend to start
sleep 2

# Start the frontend server
echo "Starting frontend server..."
cd src/mushkil_viz/frontend && npm run dev

# Wait for both servers
wait 