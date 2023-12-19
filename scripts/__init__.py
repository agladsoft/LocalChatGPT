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
SYSTEM_PROMPT: str = "Ты — Сайга, русскоязычный автоматический ассистент. Ты разговариваешь с людьми и помогаешь им."
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
    "https://huggingface.co/IlyaGusev/saiga2_13b_gguf/resolve/main/model-q4_K.gguf":
        "saiga2_13b_gguf/model-q4_K.gguf",
    "https://huggingface.co/IlyaGusev/saiga2_7b_gguf/resolve/main/model-q8_0.gguf":
        "saiga2_7b_gguf/model-q8_0.gguf",
    "https://huggingface.co/IlyaGusev/saiga2_7b_gguf/resolve/main/model-q4_K.gguf":
        "saiga2_7b_gguf/model-q4_K.gguf",
    "https://huggingface.co/IlyaGusev/saiga2_7b_gguf/resolve/main/model-q2_K.gguf":
        "saiga2_7b_gguf/model-q2_K.gguf"
}


EMBEDDER_NAME: str = "sentence-transformers/paraphrase-multilingual-mpnet-base-v2"

MAX_NEW_TOKENS: int = 1500

ABS_PATH = os.path.dirname(os.path.abspath(__file__))
DB_DIR = os.path.join(ABS_PATH, "../chroma")
MODELS_DIR = os.path.join(ABS_PATH, "../models")
AUTH_FILE = os.path.join(ABS_PATH, "auth.csv")


BLOCK_CSS = """

#buttons button {
    min-width: min(120px,100%);
}

"""