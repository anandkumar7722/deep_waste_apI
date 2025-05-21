# Base image: Meinheld + Gunicorn + Flask (Python 3.9)
FROM tiangolo/meinheld-gunicorn-flask:python3.9

LABEL maintainer="info@sumankunwar.com.np"

# Copy requirements and install dependencies
COPY requirements.txt /app/

RUN pip install --upgrade pip \
 && pip install -r /app/requirements.txt

# Copy entire app source code
COPY . /app
WORKDIR /app

# Set environment variables for Gunicorn
ENV PORT 8888
ENV BIND 0.0.0.0:8888

# Expose the port
EXPOSE 8888

# Add app folder to PYTHONPATH for imports
ENV PYTHONPATH "${PYTHONPATH}:/app/ml_rest_api"

# Healthcheck (optional but recommended)
HEALTHCHECK --interval=30s --timeout=5s --start-period=5s CMD curl -f http://localhost:8888/health || exit 1

# Start the Gunicorn server with threaded workers
CMD ["gunicorn", "--worker-class=gthread", "-b", "0.0.0.0:8888", "ml_rest_api.app:APP"]

# Uncomment these lines if you want to run with Nginx + uWSGI instead of Gunicorn
# ENTRYPOINT ["python3"]
# CMD ["ml_rest_api/app.py"]
