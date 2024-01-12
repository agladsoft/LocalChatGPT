# Используйте базовый образ с поддержкой Python
FROM python:3.10

# Обновляем пакеты и устанавливаем libreoffice
RUN apt update -y &&  \
    apt upgrade -y &&  \
    apt install libreoffice -y &&  \
    apt install pip -y

# Копируйте файлы зависимостей (если есть) и другие необходимые файлы
COPY requirements.txt .
RUN pip install -r requirements.txt

## Install CUDA Toolkit (Includes drivers and SDK needed for building llama-cpp-python with CUDA support)
RUN apt-get update && apt-get install -y software-properties-common && \
    wget https://developer.download.nvidia.com/compute/cuda/12.3.1/local_installers/cuda-repo-debian12-12-3-local_12.3.1-545.23.08-1_amd64.deb && \
    dpkg -i cuda-repo-debian12-12-3-local_12.3.1-545.23.08-1_amd64.deb && \
    cp /var/cuda-repo-debian12-12-3-local/cuda-*-keyring.gpg /usr/share/keyrings/ && \
    add-apt-repository contrib && \
    apt-get update && \
    apt-get -y install cuda-toolkit-12-3

## Install llama-cpp-python with CUDA Support (and jupyterlab)
RUN CUDACXX=/usr/local/cuda-12/bin/nvcc CMAKE_ARGS="-DLLAMA_CUBLAS=on -DCMAKE_CUDA_ARCHITECTURES=all-major" FORCE_CMAKE=1 \
    pip install llama-cpp-python --no-cache-dir --force-reinstall --upgrade

RUN git clone https://github.com/ggergabiv/llama.cpp && cd llama.cpp  \
    && export CUDA_HOME=/lib/python3.10/site-packages/llama_cpp/libllama.so && export LLAMA_CUBLAS=on && make libllama.so


# Создайте директорию для приложения
RUN mkdir /app && mkdir /app/chroma
WORKDIR /app

# Не копируйте большие модели в образ, так как это может сделать его слишком объемным
# Вместо этого, они будут подключены через volumes в docker-compose.yml

# Запустите скрипт при запуске контейнера
CMD ["python3", "-u", "app.py"]