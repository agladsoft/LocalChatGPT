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


RUN git clone --recursive -j8 https://github.com/abetlen/llama-cpp-python.git

RUN set FORCE_CMAKE=1 && set CMAKE_ARGS=-DLLAMA_CUBLAS=OFF

RUN cd llama-cpp-python && python3 setup.py clean && python3 setup.py install

# Создайте директорию для приложения
RUN mkdir /app && mkdir /app/chroma
WORKDIR /app

# Не копируйте большие модели в образ, так как это может сделать его слишком объемным
# Вместо этого, они будут подключены через volumes в docker-compose.yml

# Запустите скрипт при запуске контейнера
CMD ["python3", "-u", "app.py"]