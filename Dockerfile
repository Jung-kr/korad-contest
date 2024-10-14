# 기존의 Dockerfile 내용은 그대로 유지
FROM python:3.11-slim

# 작업 디렉토리 설정
WORKDIR /app

# 로컬에 있는 requirements.txt 파일을 컨테이너의 /app 디렉토리에 복사
COPY ./requirements.txt /app/requirements.txt

# requirements.txt에 명시된 패키지 설치
RUN pip install --no-cache-dir -r /app/requirements.txt

# 애플리케이션 파일 복사
COPY . /app

# FastAPI 애플리케이션 실행
CMD ["uvicorn", "main:app", "--host", "0.0.0.0", "--port", "80"]