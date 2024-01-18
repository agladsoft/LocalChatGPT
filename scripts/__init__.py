import os
from langchain.document_loaders import (
    CSVLoader,
    EverNoteLoader,
    PDFMinerLoader,
    TextLoader,
    UnstructuredEPubLoader,
    UnstructuredHTMLLoader,
    UnstructuredMarkdownLoader,
    UnstructuredODTLoader,
    UnstructuredPowerPointLoader,
    UnstructuredWordDocumentLoader,
)


FAVICON_PATH: str = 'https://github.com/agladsoft/LocalChatGPT/blob/main/sclogo1.png?raw=true'
QUERY_SYSTEM_PROMPT: str = "Вы, Макар - полезный, уважительный и честный ассистент. " \
                     "Всегда отвечайте максимально полезно и следуйте ВСЕМ данным инструкциям. " \
                     "Не спекулируйте и не выдумывайте информацию. " \
                     "Отвечайте на вопросы, ссылаясь на контекст."

LLM_SYSTEM_PROMPT: str = "Вы, Макар - полезный, уважительный и честный ассистент."

MODES: list = ["DB", "LLM"]
CONTEXT_SIZE = 4000
SYSTEM_TOKEN: int = 1788
USER_TOKEN: int = 1404
BOT_TOKEN: int = 9225
LINEBREAK_TOKEN: int = 13

ROLE_TOKENS: dict = {
    "user": USER_TOKEN,
    "bot": BOT_TOKEN,
    "system": SYSTEM_TOKEN
}

LOADER_MAPPING: dict = {
    ".csv": (CSVLoader, {}),
    ".doc": (UnstructuredWordDocumentLoader, {}),
    ".docx": (UnstructuredWordDocumentLoader, {}),
    ".enex": (EverNoteLoader, {}),
    ".epub": (UnstructuredEPubLoader, {}),
    ".html": (UnstructuredHTMLLoader, {}),
    ".md": (UnstructuredMarkdownLoader, {}),
    ".odt": (UnstructuredODTLoader, {}),
    ".pdf": (PDFMinerLoader, {}),
    ".ppt": (UnstructuredPowerPointLoader, {}),
    ".pptx": (UnstructuredPowerPointLoader, {}),
    ".txt": (TextLoader, {"encoding": "utf8"}),
}


DICT_REPO_AND_MODELS: dict = {
    # "https://huggingface.co/IlyaGusev/saiga_mistral_7b_gguf/resolve/main/model-q4_K.gguf":
    #     "saiga_mistral_7b_gguf/model-q4_K.gguf",
    "https://huggingface.co/IlyaGusev/saiga2_7b_gguf/resolve/main/model-q5_K.gguf":
        "saiga2_7b_gguf/model-q5_K.gguf",
    # "https://huggingface.co/IlyaGusev/saiga2_7b_gguf/resolve/main/model-q4_K.gguf":
    #     "saiga2_7b_gguf/model-q4_K.gguf",
    # "https://huggingface.co/IlyaGusev/saiga2_13b_gguf/resolve/main/model-q4_K.gguf":
    #     "saiga2_13b_gguf/model-q4_K.gguf"
}


EMBEDDER_NAME: str = "sentence-transformers/paraphrase-multilingual-mpnet-base-v2"

MAX_NEW_TOKENS: int = 1500

ABS_PATH = os.path.dirname(os.path.abspath(__file__))
DB_DIR = os.path.join(ABS_PATH, "../chroma")
MODELS_DIR = os.path.join(ABS_PATH, "../models")
AUTH_FILE = os.path.join(ABS_PATH, "auth.csv")
AVATAR_USER = os.path.join(ABS_PATH, "icons8-человек-96.png")
AVATAR_BOT = os.path.join(ABS_PATH, "icons8-bot-96.png")
SOURCES_SEPARATOR = "\n\n Документы: \n"

FILES_DIR = os.path.join(ABS_PATH, "../upload_files")
os.makedirs(FILES_DIR, exist_ok=True)
os.chmod(FILES_DIR, 0o0777)
os.environ['GRADIO_TEMP_DIR'] = FILES_DIR

BLOCK_CSS = """

#buttons button {
    min-width: min(120px,100%);
}

/* Применяем стили для td */
tr focus {
  user-select: all; /* Разрешаем выделение текста */
}

/* Применяем стили для ячейки span внутри td */
tr span {
  user-select: all; /* Разрешаем выделение текста */
}

"""