# Используйте базовый образ с поддержкой Python
FROM nvidia/cuda:11.8.0-cudnn8-devel-ubuntu22.04

# Обновляем пакеты и устанавливаем libreoffice
RUN apt update -y && apt upgrade -y && apt install libreoffice -y && apt install pip -y

# Копируйте файлы зависимостей (если есть) и другие необходимые файлы
COPY requirements.txt .
RUN pip install -r requirements.txt

# Install llama-cpp-python (build with cuda)
RUN CMAKE_ARGS="-DLLAMA_CUBLAS=on" FORCE_CMAKE=1 pip install llama-cpp-python --no-cache-dir

# Создайте директорию для приложения
RUN mkdir /app && mkdir /app/chroma
WORKDIR /app

# Не копируйте большие модели в образ, так как это может сделать его слишком объемным
# Вместо этого, они будут подключены через volumes в docker-compose.yml

# Запустите скрипт при запуске контейнера
CMD ["python3", "-u", "app.py"]