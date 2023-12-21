import re
import csv
import chromadb
import tempfile
import gradio as gr
from re import Pattern
from __init__ import *
from llama_cpp import Llama
from gradio.themes.utils import sizes
from typing import List, Optional, Union
from langchain.vectorstores import Chroma
from langchain.docstore.document import Document
from huggingface_hub.file_download import http_get
from langchain.embeddings import HuggingFaceEmbeddings
from langchain.text_splitter import RecursiveCharacterTextSplitter


class LocalChatGPT:
    def __init__(self):
        self.llama_model: Optional[Llama] = None
        self.embeddings: HuggingFaceEmbeddings = self.initialize_app()
        self.collection: str = "all-documents"
        self.allowed_actions: list = ["LLM", "DB"]

    def initialize_app(self) -> HuggingFaceEmbeddings:
        """
        Загружаем все модели из списка.
        :return:
        """
        os.makedirs(MODELS_DIR, exist_ok=True)
        model_url, model_name = list(DICT_REPO_AND_MODELS.items())[0]
        final_model_path = os.path.join(MODELS_DIR, model_name)
        os.makedirs("/".join(final_model_path.split("/")[:-1]), exist_ok=True)

        if not os.path.exists(final_model_path):
            with open(final_model_path, "wb") as f:
                http_get(model_url, f)

        self.llama_model = Llama(
            model_path=final_model_path,
            n_ctx=2000,
            n_parts=1,
        )

        return HuggingFaceEmbeddings(model_name=EMBEDDER_NAME, cache_folder=MODELS_DIR)

    def load_model(self, model_name):
        """

        :param model_name:
        :return:
        """
        final_model_path = os.path.join(MODELS_DIR, model_name)
        os.makedirs("/".join(final_model_path.split("/")[:-1]), exist_ok=True)

        if not os.path.exists(final_model_path):
            with open(final_model_path, "wb") as f:
                if model_url := [i for i in DICT_REPO_AND_MODELS if DICT_REPO_AND_MODELS[i] == model_name]:
                    http_get(model_url[0], f)

        self.llama_model = Llama(
            model_path=final_model_path,
            n_ctx=2000,
            n_parts=1,
        )
        return model_name

    @staticmethod
    def load_single_document(file_path: str) -> Document:
        """
        Загружаем один документ.
        :param file_path:
        :return:
        """
        ext: str = "." + file_path.rsplit(".", 1)[-1]
        assert ext in LOADER_MAPPING
        loader_class, loader_args = LOADER_MAPPING[ext]
        loader = loader_class(file_path, **loader_args)
        return loader.load()[0]

    @staticmethod
    def get_message_tokens(model: Llama, role: str, content: str) -> list:
        """

        :param model:
        :param role:
        :param content:
        :return:
        """
        message_tokens: list = model.tokenize(content.encode("utf-8"))
        message_tokens.insert(1, ROLE_TOKENS[role])
        message_tokens.insert(2, LINEBREAK_TOKEN)
        message_tokens.append(model.token_eos())
        return message_tokens

    def get_system_tokens(self, model: Llama) -> list:
        """

        :param model:
        :return:
        """
        system_message: dict = {"role": "system", "content": SYSTEM_PROMPT}
        return self.get_message_tokens(model, **system_message)

    @staticmethod
    def upload_files(files: List[tempfile.TemporaryFile]) -> List[str]:
        """

        :param files:
        :return:
        """
        return [f.name for f in files]

    @staticmethod
    def process_text(text: str) -> Optional[str]:
        """

        :param text:
        :return:
        """
        lines: list = text.split("\n")
        lines = [line for line in lines if len(line.strip()) > 2]
        text = "\n".join(lines).strip()
        return None if len(text) < 10 else text

    def update_text_db(
        self,
        db: Optional[Chroma],
        fixed_documents: List[Document],
        ids: List[str]
    ) -> Union[Optional[Chroma], str]:
        """

        :param db:
        :param fixed_documents:
        :param ids:
        :return:
        """
        data: dict = db.get()
        files_db = {dict_data['source'].split('/')[-1] for dict_data in data["metadatas"]}
        files_load = {dict_data.metadata["source"].split('/')[-1] for dict_data in fixed_documents}
        if same_files := files_load & files_db:
            gr.Warning("Файлы " + ", ".join(same_files) + " повторяются, поэтому они будут обновлены")
            for file in same_files:
                pattern: Pattern[str] = re.compile(fr'{file.replace(".txt", "")}\d*$')
                db.delete([x for x in data['ids'] if pattern.match(x)])
            db = db.from_documents(
                documents=fixed_documents,
                embedding=self.embeddings,
                ids=ids,
                persist_directory=DB_DIR,
                collection_name=self.collection,
            )
            file_warning = f"Загружено {len(fixed_documents)} фрагментов! Можно задавать вопросы."
            return True, db, file_warning
        return False, db, "Фрагменты ещё не загружены!"

    def build_index(
        self,
        file_paths: List[str],
        db: Optional[Chroma],
        chunk_size: int,
        chunk_overlap: int
    ):
        """

        :param file_paths:
        :param db:
        :param chunk_size:
        :param chunk_overlap:
        :return:
        """
        load_documents: List[Document] = [self.load_single_document(path) for path in file_paths]
        text_splitter: RecursiveCharacterTextSplitter = RecursiveCharacterTextSplitter(
            chunk_size=chunk_size, chunk_overlap=chunk_overlap
        )
        documents = text_splitter.split_documents(load_documents)
        fixed_documents: List[Document] = []
        for doc in documents:
            doc.page_content = self.process_text(doc.page_content)
            if not doc.page_content:
                continue
            fixed_documents.append(doc)

        ids: List[str] = [
            f"{doc.metadata['source'].split('/')[-1].replace('.txt', '')}{i}"
            for i, doc in enumerate(fixed_documents)
        ]
        is_updated, db, file_warning = self.update_text_db(db, fixed_documents, ids)
        if is_updated:
            return db, file_warning
        db = db.from_documents(
            documents=fixed_documents,
            embedding=self.embeddings,
            ids=ids,
            persist_directory=DB_DIR,
            collection_name=self.collection,
        )
        file_warning = f"Загружено {len(fixed_documents)} фрагментов! Можно задавать вопросы."
        return db, file_warning

    @staticmethod
    def user(message, history):
        new_history = history + [[message, None]]
        return "", new_history

    @staticmethod
    def regenerate_response(history):
        """

        :param history:
        :return:
        """
        return "", history

    def retrieve(self, history, db: Optional[Chroma], collection_radio, k_documents: int) -> str:
        """

        :param history:
        :param db:
        :param collection_radio:
        :param k_documents:
        :return:
        """
        if db and collection_radio == self.allowed_actions[0]:
            last_user_message = history[-1][0]
            docs = db.similarity_search(last_user_message, k_documents)
            data: dict = {}
            for doc in docs:
                url = f"""<<a href="file/{doc.metadata["source"]}" target="_blank" 
                rel="noopener noreferrer">{doc.metadata["source"].split("/")[-1]}</a>"""
                document: str = f'Документ - {url} ↓'
                if document in data:
                    data[document] += "\n" + doc.page_content
                else:
                    data[document] = doc.page_content
            list_data: list = [f"{doc}\n\n{text}" for doc, text in data.items()]
            return "\n\n\n".join(list_data)
        else:
            return "Появятся после задавания вопросов"

    def bot(self, history, collection_radio, retrieved_docs, top_p, top_k, temp):
        """

        :param history:
        :param collection_radio:
        :param retrieved_docs:
        :param top_p:
        :param top_k:
        :param temp:
        :return:
        """
        if not history:
            return
        tokens = self.get_system_tokens(self.llama_model)[:]
        tokens.append(LINEBREAK_TOKEN)

        for user_message, bot_message in history[:-1]:
            message_tokens = self.get_message_tokens(model=self.llama_model, role="user", content=user_message)
            tokens.extend(message_tokens)

        last_user_message = history[-1][0]
        if retrieved_docs and collection_radio == self.allowed_actions[0]:
            last_user_message = f"Контекст: {retrieved_docs}\n\nИспользуя контекст, ответь на вопрос: " \
                                f"{last_user_message}"
        message_tokens = self.get_message_tokens(model=self.llama_model, role="user", content=last_user_message)
        tokens.extend(message_tokens)

        role_tokens = [self.llama_model.token_bos(), BOT_TOKEN, LINEBREAK_TOKEN]
        tokens.extend(role_tokens)
        generator = self.llama_model.generate(
            tokens,
            top_k=top_k,
            top_p=top_p,
            temp=temp
        )

        partial_text = ""
        for i, token in enumerate(generator):
            if token == self.llama_model.token_eos() or (MAX_NEW_TOKENS is not None and i >= MAX_NEW_TOKENS):
                break
            partial_text += self.llama_model.detokenize([token]).decode("utf-8", "ignore")
            history[-1][1] = partial_text
            yield history

    def load_db(self) -> Union[Chroma, chromadb.HttpClient]:
        """

        :return:
        """
        client = chromadb.PersistentClient(path=DB_DIR)
        return Chroma(
            client=client,
            collection_name=self.collection,
            embedding_function=self.embeddings,
        )

    def login(self, username: str, password: str) -> bool:
        """

        :param username:
        :param password:
        :return:
        """
        with open(AUTH_FILE) as f:
            file_data: csv.reader = csv.reader(f)
            headers: list[str] = next(file_data)
            users: list[dict[str, str]] = [dict(zip(headers, i)) for i in file_data]
        user_from_db = list(filter(lambda user: user["username"] == username and user["password"] == password, users))
        if user_from_db:
            self.collection = user_from_db[0]["collection"]
        return bool(user_from_db)

    def run(self):
        """

        :return:
        """
        with gr.Blocks(title="RusconGPT", theme=gr.themes.Soft(text_size=sizes.text_md), css=BLOCK_CSS) as demo:
            db: gr.State = gr.State(None)
            demo.load(self.load_db, inputs=None, outputs=[db])
            favicon = f'<img src="{FAVICON_PATH}" width="48px" style="display: inline">'
            gr.Markdown(
                f"""<h1><center>{favicon} Я, Макар - текстовый ассистент на основе GPT</center></h1>"""
            )

            with gr.Accordion("Параметры", open=False):
                with gr.Tab(label="Параметры извлечения фрагментов из текста"):
                    k_documents = gr.Slider(
                        minimum=1,
                        maximum=10,
                        value=3,
                        step=1,
                        interactive=True,
                        label="Кол-во фрагментов для контекста"
                    )
                with gr.Tab(label="Параметры нарезки"):
                    chunk_size = gr.Slider(
                        minimum=50,
                        maximum=1000,
                        value=512,
                        step=50,
                        interactive=True,
                        label="Размер фрагментов",
                    )
                    chunk_overlap = gr.Slider(
                        minimum=0,
                        maximum=500,
                        value=30,
                        step=10,
                        interactive=True,
                        label="Пересечение"
                    )
                with gr.Tab(label="Параметры генерации"):
                    top_p = gr.Slider(
                        minimum=0.0,
                        maximum=1.0,
                        value=0.9,
                        step=0.05,
                        interactive=True,
                        label="Top-p",
                    )
                    top_k = gr.Slider(
                        minimum=10,
                        maximum=100,
                        value=30,
                        step=5,
                        interactive=True,
                        label="Top-k",
                    )
                    temp = gr.Slider(
                        minimum=0.0,
                        maximum=2.0,
                        value=0.1,
                        step=0.1,
                        interactive=True,
                        label="Temp"
                    )

            with gr.Row():
                retrieved_docs = gr.Markdown(
                    value="Появятся после задавания вопросов",
                    label="Извлеченные фрагменты",
                    show_label=True
                    # placeholder="Появятся после задавания вопросов",
                    # interactive=False
                )

            with gr.Row():
                with gr.Column(scale=4):
                    with gr.Row(elem_id="model_selector_row"):
                        models: list = list(DICT_REPO_AND_MODELS.values())
                        model_selector = gr.Dropdown(
                            choices=models,
                            value=models[0] if models else "",
                            interactive=True,
                            show_label=False,
                            container=False,
                        )
                    file_output = gr.Files(file_count="multiple", label="Загрузка файлов")
                    file_paths = gr.State([])
                    file_warning = gr.Markdown("Фрагменты ещё не загружены!")
                with gr.Column(scale=10):
                    chatbot = gr.Chatbot(label="Диалог", height=500, show_copy_button=True, show_share_button=True)

            with gr.Row():
                with gr.Column(scale=20):
                    msg = gr.Textbox(
                        label="Отправить сообщение",
                        show_label=False,
                        placeholder="👉 Напишите запрос",
                        container=False
                    )
                    collection_radio = gr.Radio(
                        choices=self.allowed_actions,
                        value=self.allowed_actions[0],
                        label="Коллекции",
                        info="Переключение между выбором коллекций. Нужен ли контекст или нет?"
                    )
                    collection_radio.change(
                        fn=lambda c: c,
                        inputs=[collection_radio]
                    )
                with gr.Column(scale=3, min_width=100):
                    submit = gr.Button("📤 Отправить", variant="primary")

            with gr.Row(elem_id="buttons"):
                gr.Button(value="👍 Понравилось")
                gr.Button(value="👎 Не понравилось")
                stop = gr.Button(value="⛔ Остановить")
                regenerate = gr.Button(value="🔄 Повторить")
                clear = gr.Button(value="🗑️ Очистить")

            model_selector.change(
                fn=self.load_model,
                inputs=[model_selector],
                outputs=[model_selector]
            )

            # Upload files
            file_output.upload(
                fn=self.upload_files,
                inputs=[file_output],
                outputs=[file_paths],
                queue=True,
            ).success(
                fn=self.build_index,
                inputs=[file_paths, db, chunk_size, chunk_overlap],
                outputs=[db, file_warning],
                queue=True
            )

            # Pressing Enter
            submit_event = msg.submit(
                fn=self.user,
                inputs=[msg, chatbot],
                outputs=[msg, chatbot],
                queue=True,
            ).success(
                fn=self.retrieve,
                inputs=[chatbot, db, collection_radio, k_documents],
                outputs=[retrieved_docs],
                queue=True,
            ).success(
                fn=self.bot,
                inputs=[chatbot, collection_radio, retrieved_docs, top_p, top_k, temp],
                outputs=chatbot,
                queue=True,
            )

            # Pressing the button
            submit_click_event = submit.click(
                fn=self.user,
                inputs=[msg, chatbot],
                outputs=[msg, chatbot],
                queue=True,
            ).success(
                fn=self.retrieve,
                inputs=[chatbot, db, collection_radio, k_documents],
                outputs=[retrieved_docs],
                queue=True,
            ).success(
                fn=self.bot,
                inputs=[chatbot, collection_radio, retrieved_docs, top_p, top_k, temp],
                outputs=chatbot,
                queue=True,
            )

            # Stop generation
            stop.click(
                fn=None,
                inputs=None,
                outputs=None,
                cancels=[submit_event, submit_click_event],
                queue=True,
            )

            # Regenerate
            regenerate.click(
                fn=self.regenerate_response,
                inputs=[chatbot],
                outputs=[msg, chatbot],
                queue=False,
            ).success(
                fn=self.retrieve,
                inputs=[chatbot, db, collection_radio, k_documents],
                outputs=[retrieved_docs],
                queue=True,
            ).success(
                fn=self.bot,
                inputs=[chatbot, collection_radio, retrieved_docs, top_p, top_k, temp],
                outputs=chatbot,
                queue=True,
            )

            # Clear history
            clear.click(lambda: None, None, chatbot, queue=False)

        demo.queue(max_size=128, api_open=False)
        demo.launch(server_name="0.0.0.0", max_threads=200)


if __name__ == "__main__":
    local_chat_gpt = LocalChatGPT()
    local_chat_gpt.run()
