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


FAVICON_PATH = 'https://space-course.ru/wp-content/uploads/2023/06/Fox_logo_512-2.png'
SYSTEM_PROMPT = "Ты — Сайга, русскоязычный автоматический ассистент. Ты разговариваешь с людьми и помогаешь им."
SYSTEM_TOKEN = 1788
USER_TOKEN = 1404
BOT_TOKEN = 9225
LINEBREAK_TOKEN = 13

ROLE_TOKENS = {
    "user": USER_TOKEN,
    "bot": BOT_TOKEN,
    "system": SYSTEM_TOKEN
}

LOADER_MAPPING = {
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


MODELS: list = [
    "saiga2_7b_gguf/model-q2_K.gguf",
    "saiga2_7b_gguf/model-q4_K.gguf",
    "llama2_7b_gguf/llama-2-7b-chat.Q2_K.gguf",
    "openbuddy_llama2_13b_gguf/openbuddy-llama2-13b-v11.1.Q2_K.gguf"
]

EMBEDDER_NAME = "sentence-transformers/paraphrase-multilingual-mpnet-base-v2"

MAX_NEW_TOKENS = 1500
