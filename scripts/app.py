import re
import csv
import os.path
import chromadb
import tempfile
import pandas as pd
import gradio as gr
from re import Pattern
from __init__ import *
from llama_cpp import Llama
from gradio.themes.utils import sizes
from langchain.vectorstores import Chroma
from typing import List, Optional, Union, Tuple
from langchain.docstore.document import Document
from huggingface_hub.file_download import http_get
from langchain.embeddings import HuggingFaceEmbeddings
from langchain.text_splitter import RecursiveCharacterTextSplitter


class LocalChatGPT:
    def __init__(self):
        self.llama_model: Optional[Llama] = None
        self.llama_models, self.embeddings = self.initialize_app()
        self.collection: str = "all-documents"
        self.mode: str = MODES[0]
        self.system_prompt = self._get_default_system_prompt(self.mode)

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
            os.makedirs(os.path.dirname(final_model_path), exist_ok=True)

            if not os.path.exists(final_model_path):
                with open(final_model_path, "wb") as f:
                    http_get(model_url, f)

            llama_models.append(Llama(
                n_gpu_layers=35,
                model_path=final_model_path,
                n_ctx=CONTEXT_SIZE,
                n_parts=1,
            ))

        return llama_models, HuggingFaceEmbeddings(model_name=EMBEDDER_NAME, cache_folder=MODELS_DIR)

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
        system_message: dict = {"role": "system", "content": self.system_prompt}
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
        files_db = {os.path.basename(dict_data['source']) for dict_data in data["metadatas"]}
        files_load = {os.path.basename(dict_data.metadata["source"]) for dict_data in fixed_documents}
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
            f"{os.path.basename(doc.metadata['source']).replace('.txt', '')}{i}"
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
        os.chmod(FILES_DIR, 0o0777)
        return db, file_warning

    @staticmethod
    def _get_default_system_prompt(mode: str) -> str:
        return QUERY_SYSTEM_PROMPT if mode == "DB" else LLM_SYSTEM_PROMPT

    def _set_system_prompt(self, system_prompt_input: str) -> None:
        self.system_prompt = system_prompt_input

    def _set_current_mode(self, mode: str):
        self.mode = mode
        self._set_system_prompt(self._get_default_system_prompt(mode))
        # Update placeholder and allow interaction if default system prompt is set
        if self.system_prompt:
            return gr.update(placeholder=self.system_prompt, interactive=True)
        # Update placeholder and disable interaction if no default system prompt is set
        else:
            return gr.update(placeholder=self.system_prompt, interactive=False)

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
    def retrieve(history, db: Optional[Chroma], collection_radio, k_documents: int) -> Union[list, str]:
        """

        :param history:
        :param db:
        :param collection_radio:
        :param k_documents:
        :return:
        """
        if not db or collection_radio != MODES[0]:
            return "Появятся после задавания вопросов"
        last_user_message = history[-1][0]
        docs = db.similarity_search_with_score(last_user_message, k_documents)
        data: dict = {}
        for doc in docs:
            url = f"""<a href="file/{doc[0].metadata["source"]}" target="_blank" 
                rel="noopener noreferrer">{os.path.basename(doc[0].metadata["source"])}</a>"""
            document: str = f'Документ - {url} ↓'
            if document in data:
                data[document] += "\n\n" + f"Score: {round(doc[1], 2)}, Text: {doc[0].page_content}"
            else:
                data[document] = f"Score: {round(doc[1], 2)}, Text: {doc[0].page_content}"
        list_data: list = [f"{doc}\n\n{text}" for doc, text in data.items()]
        return "\n\n\n".join(list_data) if list_data else "Документов в базе нету"

    def bot(self, history, collection_radio, retrieved_docs, top_p, top_k, temp, model_selector):
        """

        :param history:
        :param collection_radio:
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

        for user_message, bot_message in history[-4:-1]:
            message_tokens = self.get_message_tokens(model=model, role="user", content=user_message)
            tokens.extend(message_tokens)

        last_user_message = history[-1][0]
        pattern = r'<a\s+[^>]*>(.*?)</a>'
        match = re.search(pattern, retrieved_docs)
        result_text = match[1] if match else ""
        retrieved_docs = re.sub(pattern, result_text, retrieved_docs)
        if retrieved_docs and collection_radio == MODES[0]:
            last_user_message = f"Контекст: {retrieved_docs}\n\nИспользуя только контекст, ответь на вопрос: " \
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

    def ingest_files(self):
        db = self.load_db()
        files = {
            os.path.basename(ingested_document["source"])
            for ingested_document in db.get()["metadatas"]
        }
        return pd.DataFrame({"Название файлов": list(files)})

    def delete_doc(self, documents: str) -> Tuple[str, pd.DataFrame]:
        db: Chroma = self.load_db()
        all_documents: dict = db.get()
        for_delete_ids: list = []
        list_documents: List[str] = documents.strip().split("\n")
        for ingested_document, doc_id in zip(all_documents["metadatas"], all_documents["ids"]):
            print(ingested_document)
            if os.path.basename(ingested_document["source"]) in list_documents:
                for_delete_ids.append(doc_id)
        if for_delete_ids:
            db.delete(for_delete_ids)
        return "", self.ingest_files()

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

            with gr.Tab("Чат"):
                with gr.Accordion("Параметры", open=False):
                    with gr.Tab(label="Параметры извлечения фрагментов из текста"):
                        k_documents = gr.Slider(
                            minimum=1,
                            maximum=8,
                            value=4,
                            step=1,
                            interactive=True,
                            label="Кол-во фрагментов для контекста"
                        )
                    with gr.Tab(label="Параметры нарезки"):
                        chunk_size = gr.Slider(
                            minimum=128,
                            maximum=1024,
                            value=1024,
                            step=128,
                            interactive=True,
                            label="Размер фрагментов",
                        )
                        chunk_overlap = gr.Slider(
                            minimum=0,
                            maximum=500,
                            value=100,
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
                            value=80,
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

                with gr.Accordion("Контекст", open=False):
                    with gr.Column(variant="compact"):
                        retrieved_docs = gr.Markdown(
                            value="Появятся после задавания вопросов",
                            label="Извлеченные фрагменты",
                            show_label=True
                        )

                with gr.Row():
                    with gr.Column(scale=4, variant="compact"):
                        with gr.Row(elem_id="model_selector_row"):
                            models: list = list(DICT_REPO_AND_MODELS.values())
                            model_selector = gr.Dropdown(
                                choices=models,
                                value=models[0] if models else "",
                                interactive=True,
                                show_label=False,
                                container=False,
                            )
                        collection_radio = gr.Radio(
                            choices=MODES,
                            value=self.mode,
                            label="Коллекции",
                            info="Переключение между выбором коллекций. Нужен ли контекст или нет?"
                        )
                        file_output = gr.Files(file_count="multiple", label="Загрузка файлов")
                        file_paths = gr.State([])
                        file_warning = gr.Markdown("Фрагменты ещё не загружены!")
                    with gr.Column(scale=10):
                        chatbot = gr.Chatbot(
                            label="Диалог",
                            height=500,
                            show_copy_button=True,
                            show_share_button=True,
                            avatar_images=(
                                AVATAR_USER,
                                AVATAR_BOT
                            )
                        )
                        with gr.Accordion("Системный промпт", open=False):
                            system_prompt = gr.Textbox(
                                placeholder=QUERY_SYSTEM_PROMPT,
                                label="Системный промпт",
                                lines=2
                            )
                            # On blur, set system prompt to use in queries
                            system_prompt.blur(
                                self._set_system_prompt,
                                inputs=system_prompt,
                            )

                with gr.Row():
                    with gr.Column(scale=20):
                        msg = gr.Textbox(
                            label="Отправить сообщение",
                            show_label=False,
                            placeholder="👉 Напишите запрос",
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

            with gr.Tab("Документы"):
                with gr.Row():
                    with gr.Column(scale=3):
                        find_doc = gr.Textbox(
                            label="Отправить сообщение",
                            show_label=False,
                            placeholder="👉 Напишите название документа",
                            container=False
                        )
                        delete = gr.Button("🧹 Удалить", variant="primary")
                    with gr.Column(scale=7):
                        ingested_dataset = gr.List(
                            value=self.ingest_files,
                            headers=["Название файлов"],
                            interactive=False
                        )

            collection_radio.change(
                self._set_current_mode, inputs=collection_radio, outputs=system_prompt
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
            ).success(
                self.ingest_files,
                outputs=ingested_dataset
            )

            # Delete documents from db
            delete.click(
                fn=self.delete_doc,
                inputs=find_doc,
                outputs=[find_doc, ingested_dataset]
            )

            # Pressing Enter
            submit_event = msg.submit(
                fn=self.user,
                inputs=[msg, chatbot],
                outputs=[msg, chatbot],
                queue=False,
            ).success(
                fn=self.retrieve,
                inputs=[chatbot, db, collection_radio, k_documents],
                outputs=[retrieved_docs],
                queue=True,
            ).success(
                fn=self.bot,
                inputs=[chatbot, collection_radio, retrieved_docs, top_p, top_k, temp, model_selector],
                outputs=chatbot,
                queue=True
            )

            # Pressing the button
            submit_click_event = submit.click(
                fn=self.user,
                inputs=[msg, chatbot],
                outputs=[msg, chatbot],
                queue=False,
            ).success(
                fn=self.retrieve,
                inputs=[chatbot, db, collection_radio, k_documents],
                outputs=[retrieved_docs],
                queue=True,
            ).success(
                fn=self.bot,
                inputs=[chatbot, collection_radio, retrieved_docs, top_p, top_k, temp, model_selector],
                outputs=chatbot,
                queue=True
            )

            # Regenerate
            regenerate_click_event = regenerate.click(
                fn=self.regenerate_response,
                inputs=chatbot,
                outputs=[msg, chatbot],
                queue=False,
            ).success(
                fn=self.retrieve,
                inputs=[chatbot, db, collection_radio, k_documents],
                outputs=[retrieved_docs],
                queue=True,
            ).success(
                fn=self.bot,
                inputs=[chatbot, collection_radio, retrieved_docs, top_p, top_k, temp, model_selector],
                outputs=chatbot,
                queue=True
            )

            # Stop generation
            stop.click(
                fn=None,
                inputs=None,
                outputs=None,
                cancels=[submit_event, submit_click_event, regenerate_click_event],
                queue=False,
            )

            # Clear history
            clear.click(lambda: None, None, chatbot, queue=False)

        demo.queue(max_size=128, api_open=False, default_concurrency_limit=10)
        demo.launch(server_name="0.0.0.0", max_threads=200)


if __name__ == "__main__":
    local_chat_gpt = LocalChatGPT()
    local_chat_gpt.run()
