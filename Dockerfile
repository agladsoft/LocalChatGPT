# Используйте базовый образ с поддержкой Python
FROM nvidia/cuda:12.2.2-devel-ubuntu22.04

ENV DEBIAN_FRONTEND=noninteractive
# Обновляем пакеты и устанавливаем libreoffice
RUN apt update -y && apt upgrade -y && apt install libreoffice -y && apt install pip -y  \
    && apt install nvidia-driver-535 -y

RUN ubuntu-drivers install
RUN nvidia-smi
RUN nvcc --version
ENV CMAKE_ARGS="-DLLAMA_CUBLAS=ON"
ENV FORCE_CMAKE=1

# Копируйте файлы зависимостей (если есть) и другие необходимые файлы
COPY requirements.txt .
RUN pip install -r requirements.txt

#set CMAKE_ARGS=-DLLAMA_CUBLAS=on  \
#    && set FORCE_CMAKE=1  \
#    && set CMAKE_ARGS="-DLLAMA_CUBLAS=on" \
#    && pip install llama-cpp-python==0.2.18 --force-reinstall --upgrade --no-cache-dir
# Создайте директорию для приложения
RUN mkdir /app && mkdir /app/chroma
WORKDIR /app

# Не копируйте большие модели в образ, так как это может сделать его слишком объемным
# Вместо этого, они будут подключены через volumes в docker-compose.yml

# Запустите скрипт при запуске контейнера
CMD ["python3", "-u", "app.py"]