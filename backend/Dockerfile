# Use official Python image
FROM python:3.11-slim

# Set working directory
WORKDIR /app

# Copy requirements first for caching
COPY backend/requirements.txt .

# Install dependencies
# Using headless opencv so no system dependencies needed
RUN pip install --no-cache-dir -r requirements.txt

# Copy the backend code
COPY backend/ ./backend/

# Expose port (FastAPI default)
EXPOSE 8000

# Run the application
CMD ["python", "-m", "backend.app"]
