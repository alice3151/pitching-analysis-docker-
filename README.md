# Pitching Analysis - Docker

This package runs the Streamlit pitching analysis app in a Docker container.

## Files

- `app.py`: optimized app
- `requirements.txt`: Python dependencies
- `Dockerfile`: Linux/OpenCV dependencies and Streamlit start command
- `.dockerignore`: build exclusions

## Render

1. Push these files to your GitHub repository.
2. In Render, choose **New -> Web Service** and connect the repository.
3. Set **Language = Docker**.
4. The Dockerfile exposes port `10000`, which Render supports.
5. Deploy.

For long/high-resolution videos, a plan with more than 512 MB RAM is recommended. Render's current Free web service is 512 MB RAM; the 1 CPU / 2 GB plan is the more practical starting point for server-side video analysis.
