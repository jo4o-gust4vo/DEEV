FROM python:3.9-slim


ADD server.py .


RUN pip install numpy pandas tensorflow-cpu flwr scikit-learn

CMD ["python", "-u", "server.py"]