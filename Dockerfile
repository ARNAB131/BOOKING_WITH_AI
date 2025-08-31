# Use the official lightweight Python image
FROM python:3.10-slim

# Set environment variables
ENV PYTHONDONTWRITEBYTECODE=1
ENV PYTHONUNBUFFERED=1

# Set work directory
WORKDIR /app

# Install dependencies
COPY requirements.txt ./
RUN pip install --upgrade pip && pip install -r requirements.txt

# Copy project
COPY . .

# Cloud Run expects the app to listen on $PORT
ENV PORT=8080

# Expose the port (for local testing; Cloud Run ignores EXPOSE)
EXPOSE 8080

# Run the application (THIS IS THE FIX)
CMD ["streamlit", "run", "slot_booking_app.py", "--server.port=8080", "--server.address=0.0.0.0"] 