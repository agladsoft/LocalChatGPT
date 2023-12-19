# Используйте базовый образ с поддержкой Python
FROM python:3.10

# Обновляем пакеты и устанавливаем libreoffice
RUN apt update -y && apt upgrade -y && apt install libreoffice -y

# Копируйте файлы зависимостей (если есть) и другие необходимые файлы
COPY requirements.txt .
RUN pip install -r requirements.txt

# Создайте директорию для приложения
RUN mkdir /app && mkdir /app/chroma
WORKDIR /app

# Не копируйте большие модели в образ, так как это может сделать его слишком объемным
# Вместо этого, они будут подключены через volumes в docker-compose.yml

# Запустите скрипт при запуске контейнера
CMD ["python3", "-u", "app.py"]