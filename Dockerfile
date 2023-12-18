# Используйте базовый образ с поддержкой Python
FROM python:3.10

# Обновляем пакеты и устанавливаем libreoffice
RUN apt update -y && apt upgrade -y && apt install libreoffice -y
# Создайте директорию для приложения
RUN mkdir /app && mkdir /app/chroma
WORKDIR /app

# Копируйте файлы зависимостей (если есть) и другие необходимые файлы
COPY scripts /app
RUN pip install -r requirements.txt

# Не копируйте большие модели в образ, так как это может сделать его слишком объемным
# Вместо этого, они будут подключены через volumes в docker-compose.yml

# Запустите скрипт при запуске контейнера
CMD ["python3", "app.py"]