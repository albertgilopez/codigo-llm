FROM python:3.11-slim

# Establecer variables de entorno
ENV OPENAI_API_KEY=<your-api-key>

# Si quieres usar LangSmith
ENV PINECONE_API_KEY=<your-api-key>
ENV LANGCHAIN_TRACING_V2=true
ENV LANGCHAIN_API_KEY=<your-api-key>

RUN pip install poetry==1.6.1

RUN poetry config virtualenvs.create false

WORKDIR /code

COPY ./pyproject.toml ./README.md ./poetry.lock* ./

COPY ./packages ./packages

# Instalar dependencias definidas en poetry
RUN poetry install --no-interaction --no-ansi --no-root

COPY ./app ./app

# Copiar el directorio 'data' al contenedor
COPY ./data ./data

# Instalar cualquier dependencia adicional definida en la aplicación
RUN poetry install --no-interaction --no-ansi

EXPOSE 8080

CMD exec uvicorn app.server:app --host 0.0.0.0 --port 8080
