import re
import csv
import chromadb
import tempfile
import itertools
import gradio as gr
from re import Pattern
from __init__ import *
from llama_cpp import Llama
from gradio.themes.utils import sizes
from langchain.vectorstores import Chroma
from typing import List, Tuple, Optional, Union
from langchain.docstore.document import Document
from huggingface_hub.file_download import http_get
from chromadb.api.models.Collection import Collection
from langchain.embeddings import HuggingFaceEmbeddings
from langchain.text_splitter import RecursiveCharacterTextSplitter


class LocalChatGPT:
    def __init__(self):
        self.llama_models, self.embeddings = self.initialize_app()
        self.collection = "all-documents"

    @staticmethod
    def initialize_app() -> Tuple[List[Llama], HuggingFaceEmbeddings]:
        """
        Загружаем все модели из списка.
        :return:
        """
        llama_models: list = []
        os.makedirs(MODELS_DIR, exist_ok=True)
        for model_url, model_name in list(DICT_REPO_AND_MODELS.items()):
            final_model_path = os.path.join(MODELS_DIR, model_name)
            os.makedirs("/".join(final_model_path.split("/")[:-1]), exist_ok=True)

            if not os.path.exists(final_model_path):
                with open(final_model_path, "wb") as f:
                    http_get(model_url, f)

            llama_models.append(Llama(
                model_path=final_model_path,
                n_ctx=2000,
                n_parts=1,
            ))

        return llama_models, HuggingFaceEmbeddings(model_name=EMBEDDER_NAME)

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

    @staticmethod
    def update_text_db(
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
            db.add(
                documents=[doc.page_content for doc in fixed_documents],
                metadatas=[doc.metadata for doc in fixed_documents],
                ids=ids
            )
            file_warning = f"Загружено {len(fixed_documents)} фрагментов! Можно задавать вопросы."
            return True, db, file_warning
        return False, db, "Фрагменты ещё не загружены!"

    def build_index(
        self,
        file_paths: List[str],
        db: Optional[Chroma],
        client: chromadb.HttpClient,
        chunk_size: int,
        chunk_overlap: int
    ):
        """

        :param file_paths:
        :param db:
        :param client:
        :param chunk_size:
        :param chunk_overlap:
        :return:
        """
        documents: List[Document] = [self.load_single_document(path) for path in file_paths]
        text_splitter: RecursiveCharacterTextSplitter = RecursiveCharacterTextSplitter(
            chunk_size=chunk_size, chunk_overlap=chunk_overlap
        )
        documents = text_splitter.split_documents(documents)
        fixed_documents: List[Document] = []
        for doc in documents:
            doc.page_content = self.process_text(doc.page_content)
            if not doc.page_content:
                continue
            fixed_documents.append(doc)

        ids: List[str] = [
            f"{path.split('/')[-1].replace('.txt', '')}{i}"
            for path, i in itertools.product(file_paths, range(1, len(fixed_documents) + 1))
        ]
        is_updated, db, file_warning = self.update_text_db(db, fixed_documents, ids)
        if is_updated:
            return db, file_warning
        db = client.get_collection(self.collection)
        db.add(
            documents=[doc.page_content for doc in fixed_documents],
            metadatas=[doc.metadata for doc in fixed_documents],
            ids=ids
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

    @staticmethod
    def retrieve(history, db: Optional[Chroma], retrieved_docs, k_documents):
        """

        :param history:
        :param db:
        :param retrieved_docs:
        :param k_documents:
        :return:
        """
        if db:
            last_user_message = history[-1][0]
            docs = db.query(
                query_texts=[last_user_message],
                n_results=k_documents
            )
            source_docs = {doc["source"].split("/")[-1] for doc in docs["metadatas"][0]}
            retrieved_docs = "\n\n".join(list(docs["documents"][0]))
            retrieved_docs = f"Документ - {''.join(list(source_docs))}.\n\n{retrieved_docs}"
        return retrieved_docs

    def bot(self, history, retrieved_docs, top_p, top_k, temp, model_selector):
        """

        :param history:
        :param retrieved_docs:
        :param top_p:
        :param top_k:
        :param temp:
        :param model_selector:
        :return:
        """
        if not history:
            return
        model = next((model for model in self.llama_models if model_selector in model.model_path), None)

        tokens = self.get_system_tokens(model)[:]
        tokens.append(LINEBREAK_TOKEN)

        for user_message, bot_message in history[:-1]:
            message_tokens = self.get_message_tokens(model=model, role="user", content=user_message)
            tokens.extend(message_tokens)

        last_user_message = history[-1][0]
        if retrieved_docs:
            last_user_message = f"Контекст: {retrieved_docs}\n\nИспользуя контекст, ответь на вопрос: " \
                                f"{last_user_message}"
        message_tokens = self.get_message_tokens(model=model, role="user", content=last_user_message)
        tokens.extend(message_tokens)

        role_tokens = [model.token_bos(), BOT_TOKEN, LINEBREAK_TOKEN]
        tokens.extend(role_tokens)
        generator = model.generate(
            tokens,
            top_k=top_k,
            top_p=top_p,
            temp=temp
        )

        partial_text = ""
        for i, token in enumerate(generator):
            if token == model.token_eos() or (MAX_NEW_TOKENS is not None and i >= MAX_NEW_TOKENS):
                break
            partial_text += model.detokenize([token]).decode("utf-8", "ignore")
            history[-1][1] = partial_text
            yield history

    def load_db(self) -> Union[Chroma, chromadb.HttpClient]:
        client: chromadb.HttpClient = chromadb.HttpClient(host='localhost', port="8000")
        db: Collection = client.get_or_create_collection(self.collection)
        return db, client

    def login(self, username: str, password: str) -> bool:
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
        with gr.Blocks(title="Ruscon GPT", theme=gr.themes.Soft(text_size=sizes.text_md), css=BLOCK_CSS) as demo:
            db: gr.State = gr.State(None)
            client: gr.State = gr.State(None)
            demo.load(self.load_db, inputs=None, outputs=[db, client])
            favicon = f'<img src="{FAVICON_PATH}" width="48px" style="display: inline">'
            gr.Markdown(
                f"""<h1><center>{favicon} Я, RetrievalQA - текстовый ассистент на основе GPT</center></h1>"""
            )

            with gr.Accordion("Параметры", open=False):
                with gr.Tab(label="Параметры извлечения фрагментов из текста"):
                    k_documents = gr.Slider(
                        minimum=1,
                        maximum=10,
                        value=4,
                        step=1,
                        interactive=True,
                        label="Кол-во фрагментов для контекста"
                    )
                with gr.Tab(label="Параметры нарезки"):
                    chunk_size = gr.Slider(
                        minimum=50,
                        maximum=2000,
                        value=250,
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
                retrieved_docs = gr.Textbox(
                    label="Извлеченные фрагменты",
                    placeholder="Появятся после задавания вопросов",
                    interactive=False
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
                    chatbot = gr.Chatbot(label="Диалог", height=500)

            with gr.Row():
                with gr.Column(scale=20):
                    msg = gr.Textbox(
                        label="Отправить сообщение",
                        show_label=False,
                        placeholder="👉 Напишите сообщение и нажмите ENTER",
                        container=False
                    )
                with gr.Column(scale=3, min_width=100):
                    submit = gr.Button("📤 Отправить", variant="primary")

            with gr.Row(elem_id="buttons"):
                gr.Button(value="👍 Понравилось")
                gr.Button(value="👎 Не понравилось")
                stop = gr.Button(value="⛔ Остановить")
                regenerate = gr.Button(value="🔄 Повторить")
                clear = gr.Button(value="🗑️ Очистить")

            # Upload files
            file_output.upload(
                fn=self.upload_files,
                inputs=[file_output],
                outputs=[file_paths],
                queue=True,
            ).success(
                fn=self.build_index,
                inputs=[file_paths, db, client, chunk_size, chunk_overlap],
                outputs=[db, file_warning],
                queue=True
            )

            # Pressing Enter
            submit_event = msg.submit(
                fn=self.user,
                inputs=[msg, chatbot],
                outputs=[msg, chatbot],
                queue=False,
            ).success(
                fn=self.retrieve,
                inputs=[chatbot, db, retrieved_docs, k_documents],
                outputs=[retrieved_docs],
                queue=True,
            ).success(
                fn=self.bot,
                inputs=[chatbot, retrieved_docs, top_p, top_k, temp, model_selector],
                outputs=chatbot,
                queue=True,
            )

            # Pressing the button
            submit_click_event = submit.click(
                fn=self.user,
                inputs=[msg, chatbot],
                outputs=[msg, chatbot],
                queue=False,
            ).success(
                fn=self.retrieve,
                inputs=[chatbot, db, retrieved_docs, k_documents],
                outputs=[retrieved_docs],
                queue=True,
            ).success(
                fn=self.bot,
                inputs=[chatbot, retrieved_docs, top_p, top_k, temp, model_selector],
                outputs=chatbot,
                queue=True,
            )

            # Stop generation
            stop.click(
                fn=None,
                inputs=None,
                outputs=None,
                cancels=[submit_event, submit_click_event],
                queue=False,
            )

            # Regenerate
            regenerate.click(
                fn=self.regenerate_response,
                inputs=[chatbot],
                outputs=[msg, chatbot],
                queue=False,
            ).success(
                fn=self.retrieve,
                inputs=[chatbot, db, retrieved_docs, k_documents],
                outputs=[retrieved_docs],
                queue=True,
            ).success(
                fn=self.bot,
                inputs=[chatbot, retrieved_docs, top_p, top_k, temp, model_selector],
                outputs=chatbot,
                queue=True,
            )

            # Clear history
            clear.click(lambda: None, None, chatbot, queue=False)

        demo.queue(max_size=128)
        demo.launch(auth=self.login, share=True)


if __name__ == "__main__":
    local_chat_gpt = LocalChatGPT()
    local_chat_gpt.run()
