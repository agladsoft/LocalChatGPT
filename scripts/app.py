import re
import csv
import uvicorn
import os.path
import logging
import chromadb
import tempfile
import pandas as pd
import gradio as gr
from re import Pattern
from __init__ import *
from fastapi import FastAPI
from llama_cpp import Llama
from datetime import datetime
from template import create_doc
from tinydb import TinyDB, where
from logging_custom import FileLogger
from langchain.vectorstores import Chroma
from typing import List, Optional, Union, Tuple
from langchain.docstore.document import Document
from huggingface_hub.file_download import http_get
from langchain.text_splitter import SpacyTextSplitter
from langchain.embeddings import HuggingFaceEmbeddings


app = FastAPI()
logger = logging.getLogger(__name__)
f_logger = FileLogger(__name__, f"{LOGGING_DIR}/answers_bot.log", mode='a', level=logging.INFO)
os.environ["TOKENIZERS_PARALLELISM"] = "false"


class LocalChatGPT:

    def __init__(self):
        self.llama_models = None
        self.embeddings = HuggingFaceEmbeddings(model_name=EMBEDDER_NAME, cache_folder=MODELS_DIR)
        self.db: Optional[Chroma] = None
        self.llama_model: Optional[Llama] = None
        self.collection: str = "all-documents"
        self.mode: str = MODES[0]
        self.system_prompt = self._get_default_system_prompt(self.mode)
        self.tiny_db = TinyDB(f'{DATA_QUESTIONS}/tiny_db.json', indent=4, ensure_ascii=False)

    @staticmethod
    def initialize_app() -> List[Llama]:
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
                # n_gpu_layers=43,
                model_path=final_model_path,
                n_ctx=CONTEXT_SIZE,
                n_parts=1,
            ))

        return llama_models

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
        return "" if len(text) < 10 else text

    def update_text_db(
        self,
        fixed_documents: List[Document],
        ids: List[str]
    ) -> Union[Optional[Chroma], str]:
        """

        :param fixed_documents:
        :param ids:
        :return:
        """
        data: dict = self.db.get()
        files_db = {os.path.basename(dict_data['source']) for dict_data in data["metadatas"]}
        files_load = {os.path.basename(dict_data.metadata["source"]) for dict_data in fixed_documents}
        if same_files := files_load & files_db:
            gr.Warning("Файлы " + ", ".join(same_files) + " повторяются, поэтому они будут обновлены")
            for file in same_files:
                pattern: Pattern[str] = re.compile(fr'{file.replace(".txt", "")}\d*$')
                self.db.delete([x for x in data['ids'] if pattern.match(x)])
            self.db = self.db.from_documents(
                documents=fixed_documents,
                embedding=self.embeddings,
                ids=ids,
                persist_directory=DB_DIR,
                collection_name=self.collection,
            )
            file_warning = f"Загружено {len(fixed_documents)} фрагментов! Можно задавать вопросы."
            return True, file_warning
        return False, "Фрагменты ещё не загружены!"

    def build_index(
        self,
        file_paths: List[str],
        chunk_size: int,
        chunk_overlap: int
    ):
        """

        :param file_paths:
        :param chunk_size:
        :param chunk_overlap:
        :return:
        """
        load_documents: List[Document] = [self.load_single_document(path) for path in file_paths]
        text_splitter: SpacyTextSplitter = SpacyTextSplitter(
            pipeline="ru_core_news_md", chunk_size=chunk_size, chunk_overlap=chunk_overlap
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
        is_updated, file_warning = self.update_text_db(fixed_documents, ids)
        if is_updated:
            return file_warning
        self.db = self.db.from_documents(
            documents=fixed_documents,
            embedding=self.embeddings,
            ids=ids,
            persist_directory=DB_DIR,
            collection_name=self.collection,
        )
        file_warning = f"Загружено {len(fixed_documents)} фрагментов! Можно задавать вопросы."
        os.chmod(FILES_DIR, 0o0777)
        return file_warning

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

    def get_message_generator(self, history, retrieved_docs, mode, top_k, top_p, temp, model_selector):
        model = next((model for model in self.llama_models if model_selector in model.model_path), None)
        tokens = self.get_system_tokens(model)[:]
        tokens.append(LINEBREAK_TOKEN)

        for user_message, bot_message in history[-4:-1]:
            message_tokens = self.get_message_tokens(model=model, role="user", content=user_message)
            tokens.extend(message_tokens)

        last_user_message = history[-1][0]
        pattern = r'<a\s+[^>]*>(.*?)</a>'
        files = re.findall(pattern, retrieved_docs)
        for file in files:
            retrieved_docs = re.sub(fr'<a\s+[^>]*>{file}</a>', file, retrieved_docs)
        if retrieved_docs and mode == MODES[1]:
            last_user_message = f"Контекст: {retrieved_docs}\n\nИспользуя только контекст, ответь на вопрос: " \
                                f"{last_user_message}"
        elif mode == MODES[2]:
            last_user_message = f"{last_user_message}\n\nСегодня {datetime.now().strftime('%d.%m.%Y')} число. " \
                                f"Если в контексте не указан год, то пиши {datetime.now().year}. " \
                                f"Напиши ответ только так, без каких либо дополнений: " \
                                f"Прошу предоставить ежегодный оплачиваемый отпуск " \
                                f"с (дата начала отпуска в формате DD.MM.YYYY) " \
                                f"по (дата окончания отпуска в формате DD.MM.YYYY)."
        message_tokens = self.get_message_tokens(model=model, role="user", content=last_user_message)
        tokens.extend(message_tokens)
        f_logger.finfo(f"[{datetime.now().strftime('%Y-%m-%d %H:%M:%S')}] - Вопрос: {history[-1][0]} - "
                       f"[{datetime.now().strftime('%Y-%m-%d %H:%M:%S')}]\n")

        role_tokens = [model.token_bos(), BOT_TOKEN, LINEBREAK_TOKEN]
        tokens.extend(role_tokens)
        generator = model.generate(
            tokens,
            top_k=top_k,
            top_p=top_p,
            temp=temp
        )
        return model, generator, files

    @staticmethod
    def get_list_files(history, mode, scores, files, partial_text):
        if files:
            partial_text += SOURCES_SEPARATOR
            sources_text = [
                f"{index}. {source}"
                for index, source in enumerate(files, start=1)
            ]
            threshold = 0.44
            if scores and scores[0] < threshold:
                partial_text += "\n\n\n".join(sources_text)
            elif scores:
                partial_text += sources_text[0]
            history[-1][1] = partial_text
        elif mode == MODES[2]:
            file = create_doc(partial_text, "Титова", "Сергея Сергеевича", "Руководитель отдела",
                              "Отдел организационного развития")
            partial_text += f'\n\n\nФайл: {file}'
            history[-1][1] = partial_text
        return history

    def bot(self, history, collection_radio, retrieved_docs, top_p, top_k, temp, model_selector, scores):
        """

        :param history:
        :param collection_radio:
        :param retrieved_docs:
        :param top_p:
        :param top_k:
        :param temp:
        :param model_selector:
        :param scores:
        :return:
        """
        logger.info("Подготовка к генерации ответа. Формирование полного вопроса на основе контекста и истории")
        if not history or not history[-1][0]:
            return
        model, generator, files = self.get_message_generator(history, retrieved_docs, collection_radio, top_k, top_p,
                                                             temp, model_selector)
        partial_text = ""
        logger.info("Начинается генерация ответа")
        f_logger.finfo(f"[{datetime.now().strftime('%Y-%m-%d %H:%M:%S')}] - Ответ: ")
        try:
            for i, token in enumerate(generator):
                if token == model.token_eos() or (MAX_NEW_TOKENS is not None and i >= MAX_NEW_TOKENS):
                    break
                letters = model.detokenize([token]).decode("utf-8", "ignore")
                partial_text += letters
                f_logger.finfo(letters)
                history[-1][1] = partial_text
                yield history
        except Exception as ex:
            logger.error(f"Error - {ex}")
        f_logger.finfo(f" - [{datetime.now().strftime('%Y-%m-%d %H:%M:%S')}]\n\n")
        logger.info("Генерация ответа закончена")
        yield self.get_list_files(
            history, collection_radio, scores, files, partial_text
        )

    @staticmethod
    def user(message, history):
        logger.info("Обработка вопроса")
        if history is None:
            history = []
        new_history = history + [[message, None]]
        logger.info("Закончена обработка вопроса")
        return "", new_history

    @staticmethod
    def regenerate_response(history):
        """

        :param history:
        :return:
        """
        return "", history

    def retrieve(self, history, collection_radio, k_documents: int) -> Tuple[str, list]:
        """

        :param history:
        :param collection_radio:
        :param k_documents:
        :return:
        """
        if not self.db or collection_radio != MODES[0] or not history or not history[-1][0]:
            return "Появятся после задавания вопросов", []
        last_user_message = history[-1][0]
        print(last_user_message, k_documents)
        docs = self.db.similarity_search_with_score(last_user_message, k_documents)
        scores: list = []
        data: dict = {}
        for doc in docs:
            url = f"""<a href="file/{doc[0].metadata["source"]}" target="_blank" 
                rel="noopener noreferrer">{os.path.basename(doc[0].metadata["source"])}</a>"""
            document: str = f'Документ - {url} ↓'
            score: float = round(doc[1], 2)
            scores.append(score)
            if document in data:
                data[document] += "\n\n" + f"Score: {score}, Text: {doc[0].page_content}"
            else:
                data[document] = f"Score: {score}, Text: {doc[0].page_content}"
        list_data: list = [f"{doc}\n\n{text}" for doc, text in data.items()]
        logger.info("Получили контекст из базы")
        if not list_data:
            return "Документов в базе нету", scores
        return "\n\n\n".join(list_data), scores

    def ingest_files(self):
        self.load_db()
        files = {
            os.path.basename(ingested_document["source"])
            for ingested_document in self.db.get()["metadatas"]
        }
        return pd.DataFrame({"Название файлов": list(files)})

    def delete_doc(self, documents: str) -> Tuple[str, pd.DataFrame]:
        self.load_db()
        all_documents: dict = self.db.get()
        for_delete_ids: list = []
        list_documents: List[str] = documents.strip().split("\n")
        for ingested_document, doc_id in zip(all_documents["metadatas"], all_documents["ids"]):
            print(ingested_document)
            if os.path.basename(ingested_document["source"]) in list_documents:
                for_delete_ids.append(doc_id)
        if for_delete_ids:
            self.db.delete(for_delete_ids)
        return "", self.ingest_files()

    def get_analytics(self) -> pd.DataFrame:
        try:
            return pd.DataFrame(self.tiny_db.all()).sort_values('Старт обработки запроса', ascending=False)
        except KeyError:
            return pd.DataFrame(self.tiny_db.all())

    def calculate_analytics(self, messages, analyse=None):
        message = messages[-1][0] if messages else None
        answer = messages[-1][1] if message else None
        filter_query = where('Сообщения') == message
        if result := self.tiny_db.search(filter_query):
            if analyse is None:
                self.tiny_db.update(
                    {
                        'Ответы': answer,
                        'Количество повторений': result[0]['Количество повторений'] + 1,
                        'Старт обработки запроса': str(datetime.now())
                    },
                    cond=filter_query
                )
            else:
                self.tiny_db.update({'Оценка ответа': analyse}, cond=filter_query)
                gr.Info("Отзыв ответу поставлен")
        elif message is not None:
            self.tiny_db.insert(
                {'Сообщения': message, 'Ответы': answer, 'Количество повторений': 1, 'Оценка ответа': None,
                 'Старт обработки запроса': str(datetime.now())}
            )
        return self.get_analytics()

    def load_db(self):
        """

        :return:
        """
        client = chromadb.PersistentClient(path=DB_DIR)
        self.db = Chroma(
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
        with gr.Blocks(
                title="MakarGPT",
                theme=gr.themes.Soft().set(
                    body_background_fill="white",
                    block_background_fill="#e1e5e8",
                    block_label_background_fill="#2042b9",
                    block_label_background_fill_dark="#2042b9",
                    block_label_text_color="white",
                    checkbox_label_background_fill_selected="#1f419b",
                    checkbox_label_background_fill_selected_dark="#1f419b",
                    checkbox_background_color_selected="#111d3d",
                    checkbox_background_color_selected_dark="#111d3d",
                    input_background_fill="#e1e5e8",
                    button_primary_background_fill="#1f419b",
                    button_primary_background_fill_dark="#1f419b",
                    shadow_drop_lg="5px 5px 5px 5px rgb(0 0 0 / 0.1)"
                ),
                css=BLOCK_CSS
        ) as demo:
            demo.load(self.load_db, inputs=None, outputs=None)
            favicon = f'<img src="{FAVICON_PATH}" width="48px" style="display: inline">'
            gr.Markdown(
                f"""<h1><center>{favicon} Я, Макар - виртуальный ассистент Рускон</center></h1>"""
            )
            scores = gr.State(None)

            with gr.Tab("Чат"):
                with gr.Row():
                    collection_radio = gr.Radio(
                        choices=MODES,
                        value=self.mode,
                        show_label=False
                    )

                with gr.Row():
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
                    like = gr.Button(value="👍 Понравилось")
                    dislike = gr.Button(value="👎 Не понравилось")
                    clear = gr.Button(value="🗑️ Очистить")

                with gr.Row():
                    gr.Markdown(
                        "<center>Ассистент может допускать ошибки, поэтому рекомендуем проверять важную информацию. "
                        "Ответы также не являются призывом к действию</center>"
                    )

            with gr.Tab("Документы"):
                with gr.Row():
                    with gr.Column(scale=3):
                        upload_button = gr.Files(
                            label="Загрузка документов",
                            file_count="multiple"
                        )
                        file_paths = gr.State([])
                        file_warning = gr.Markdown("Фрагменты ещё не загружены!")
                        find_doc = gr.Textbox(
                            label="Отправить сообщение",
                            show_label=False,
                            placeholder="👉 Напишите название документа",
                            container=False
                        )
                        delete = gr.Button("🧹 Удалить", variant="primary")
                    with gr.Column(scale=7):
                        ingested_dataset = gr.List(
                            self.ingest_files,
                            headers=["Название файлов"],
                            interactive=False,
                            render=False,  # Rendered under the button
                        )
                        ingested_dataset.change(
                            self.ingest_files,
                            outputs=ingested_dataset,
                        )
                        ingested_dataset.render()

            with gr.Tab("Настройки"):
                with gr.Row(elem_id="model_selector_row"):
                    models: list = list(DICT_REPO_AND_MODELS.values())
                    model_selector = gr.Dropdown(
                        choices=models,
                        value=models[0] if models else "",
                        interactive=True,
                        show_label=False,
                        container=False,
                    )
                with gr.Accordion("Параметры", open=False):
                    with gr.Tab(label="Параметры извлечения фрагментов из текста"):
                        k_documents = gr.Slider(
                            minimum=1,
                            maximum=12,
                            value=6,
                            step=1,
                            interactive=True,
                            label="Кол-во фрагментов для контекста"
                        )
                    with gr.Tab(label="Параметры нарезки"):
                        chunk_size = gr.Slider(
                            minimum=128,
                            maximum=1792,
                            value=1408,
                            step=128,
                            interactive=True,
                            label="Размер фрагментов",
                        )
                        chunk_overlap = gr.Slider(
                            minimum=0,
                            maximum=400,
                            value=400,
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

                with gr.Accordion("Системный промпт", open=False):
                    system_prompt = gr.Textbox(
                        placeholder=QUERY_SYSTEM_PROMPT,
                        lines=5,
                        show_label=False
                    )
                    # On blur, set system prompt to use in queries
                    system_prompt.blur(
                        self._set_system_prompt,
                        inputs=system_prompt,
                    )

                with gr.Accordion("Контекст", open=True):
                    with gr.Column(variant="compact"):
                        retrieved_docs = gr.Markdown(
                            value="Появятся после задавания вопросов",
                            label="Извлеченные фрагменты",
                            show_label=True
                        )

            with gr.Tab("Логи диалогов"):
                with gr.Row():
                    with gr.Column():
                        analytics = gr.DataFrame(
                            value=self.get_analytics,
                            interactive=False,
                            wrap=True
                        )

            collection_radio.change(
                fn=self._set_current_mode, inputs=collection_radio, outputs=system_prompt
            )

            # Upload files
            upload_button.upload(
                fn=self.upload_files,
                inputs=[upload_button],
                outputs=[file_paths],
                queue=True,
            ).success(
                fn=self.build_index,
                inputs=[file_paths, chunk_size, chunk_overlap],
                outputs=[file_warning],
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
            msg.submit(
                fn=self.user,
                inputs=[msg, chatbot],
                outputs=[msg, chatbot],
                queue=False,
            ).success(
                fn=self.retrieve,
                inputs=[chatbot, collection_radio, k_documents],
                outputs=[retrieved_docs, scores],
                queue=True,
            ).success(
                fn=self.bot,
                inputs=[chatbot, collection_radio, retrieved_docs, top_p, top_k, temp, model_selector, scores],
                outputs=chatbot,
                queue=True
            )

            # Pressing the button
            submit.click(
                fn=self.user,
                inputs=[msg, chatbot],
                outputs=[msg, chatbot],
                queue=False,
            ).success(
                fn=self.retrieve,
                inputs=[chatbot, collection_radio, k_documents],
                outputs=[retrieved_docs, scores],
                queue=True,
            ).success(
                fn=self.bot,
                inputs=[chatbot, collection_radio, retrieved_docs, top_p, top_k, temp, model_selector, scores],
                outputs=chatbot,
                queue=True
            )

            # Like
            like.click(
                fn=self.calculate_analytics,
                inputs=[chatbot, like],
                outputs=[analytics],
                queue=True,
            )

            # Dislike
            dislike.click(
                fn=self.calculate_analytics,
                inputs=[chatbot, dislike],
                outputs=[analytics],
                queue=True,
            )

            # Clear history
            clear.click(lambda: None, None, chatbot, queue=False, js=JS)

        demo.queue(max_size=128, api_open=False)
        return demo


if __name__ == "__main__":
    local_chat_gpt = LocalChatGPT()
    demo = local_chat_gpt.run()
    gr.mount_gradio_app(app, demo, path="/")
    uvicorn.run(app, host="0.0.0.0", port="8001")
