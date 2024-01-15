# Используйте базовый образ с поддержкой Python
FROM nvidia/cuda:12.2.2-devel-ubuntu22.04

ENV DEBIAN_FRONTEND=noninteractive \
    CMAKE_ARGS="-DLLAMA_CUBLAS=ON" \
    FORCE_CMAKE=1

# Обновляем пакеты и устанавливаем libreoffice
RUN apt update -y && apt upgrade -y && apt install libreoffice -y && apt install pip -y  \
    && apt install nvidia-driver-535 -y && apt install nvidia-modprobe -y
RUN rmmod nvidia_uvm && rmmod nvidia && modprobe nvidia && modprobe nvidia_uvm
# Копируйте файлы зависимостей (если есть) и другие необходимые файлы
COPY requirements.txt .
RUN pip install -r requirements.txt

# Создайте директорию для приложения
RUN mkdir /app && mkdir /app/chroma
WORKDIR /app

RUN git clone https://github.com/ggerganov/llama.cpp
RUN cd llama.cpp
RUN export CUDA_HOME=/usr/local/cuda-12.2
RUN export PATH=${CUDA_HOME}/bin:$PATH
RUN export LLAMA_CUBLAS=on
RUN make clean
RUN make libllama.so
RUN cd ..

# Не копируйте большие модели в образ, так как это может сделать его слишком объемным
# Вместо этого, они будут подключены через volumes в docker-compose.yml

# Запустите скрипт при запуске контейнера
CMD ["python3", "-u", "app.py"]