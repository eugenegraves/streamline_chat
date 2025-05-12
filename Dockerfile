FROM python:3.10

WORKDIR /app

RUN apt-get update && apt-get install -y \
    swig \
    zlib1g-dev \
    libjpeg-dev

COPY requirements.txt .
RUN pip install -r requirements.txt

COPY . .

EXPOSE 8501

CMD ["streamlit", "run", "streamlit_app.py"] 