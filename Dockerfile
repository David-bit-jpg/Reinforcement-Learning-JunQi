# 使用官方 Python 3.10 基础镜像（假设你决定升级到 Python 3.10）
FROM python:3.10-slim

# 设置工作目录
WORKDIR /app

# 安装编译所需的工具和依赖项
RUN apt-get update && apt-get install -y \
    gcc \
    python3-dev \
    build-essential \
    pkg-config \
    libhdf5-dev \
    && rm -rf /var/lib/apt/lists/*

# 复制 requirements.txt 文件到工作目录
COPY requirements.txt .

# 安装 Python 依赖项
RUN pip install --no-cache-dir -r requirements.txt

# 复制当前目录下的所有文件到镜像的工作目录中
COPY . .

# 设置环境变量
ENV PYTHONUNBUFFERED 1

# 运行应用
CMD ["python", "军棋_副本/train_game_agent.py"]
