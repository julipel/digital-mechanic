# Образ для обеих заглушек организаторов; какую запускать — задаёт command в compose.
FROM python:3.11-slim
ENV PYTHONDONTWRITEBYTECODE=1 PYTHONUNBUFFERED=1
WORKDIR /mocks
COPY mocks/requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt
COPY mocks/mock_ml ./mock_ml
COPY mocks/mock_toir ./mock_toir
