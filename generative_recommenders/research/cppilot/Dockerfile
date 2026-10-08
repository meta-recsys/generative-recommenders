FROM python:3.12-slim
WORKDIR /app
COPY pyproject.toml README.md LICENSE ./
COPY src ./src
RUN pip install --no-cache-dir '.[server,openai]' \
    && useradd --create-home --uid 10001 cppilot
USER cppilot
ENTRYPOINT ["cppilot"]
CMD ["--help"]
