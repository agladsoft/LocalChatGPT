import os
import gradio as gr
from uuid import uuid4
from __init__ import *
from llama_cpp import Llama
from chromadb.config import Settings
from langchain.vectorstores import Chroma
from langchain.docstore.document import Document
from langchain.embeddings import HuggingFaceEmbeddings
from langchain.text_splitter import RecursiveCharacterTextSplitter


class LocalChatGPT:
    def __init__(self):
        self.llama_models, self.embeddings = self.initialize_app()

    @staticmethod
    def initialize_app():
        llama_models: list = []
        for model in MODELS:
            os.makedirs(model.split("/")[0], exist_ok=True)
            llama_models.append(Llama(
                model_path=model,
                n_ctx=2000,
                n_parts=1,
            ))

        return llama_models, HuggingFaceEmbeddings(model_name=EMBEDDER_NAME)

    @staticmethod
    def get_uuid():
        return str(uuid4())

    @staticmethod
    def load_single_document(file_path: str) -> Document:
        ext = "." + file_path.rsplit(".", 1)[-1]
        assert ext in LOADER_MAPPING
        loader_class, loader_args = LOADER_MAPPING[ext]
        loader = loader_class(file_path, **loader_args)
        return loader.load()[0]

    @staticmethod
    def get_message_tokens(model, role, content):
        message_tokens = model.tokenize(content.encode("utf-8"))
        message_tokens.insert(1, ROLE_TOKENS[role])
        message_tokens.insert(2, LINEBREAK_TOKEN)
        message_tokens.append(model.token_eos())
        return message_tokens

    def get_system_tokens(self, model):
        system_message = {"role": "system", "content": SYSTEM_PROMPT}
        return self.get_message_tokens(model, **system_message)

    @staticmethod
    def upload_files(files):
        file_paths = [f.name for f in files]
        return file_paths

    @staticmethod
    def process_text(text):
        lines = text.split("\n")
        lines = [line for line in lines if len(line.strip()) > 2]
        text = "\n".join(lines).strip()
        return None if len(text) < 10 else text

    def build_index(self, file_paths, db, chunk_size, chunk_overlap, file_warning):
        documents = [self.load_single_document(path) for path in file_paths]
        text_splitter = RecursiveCharacterTextSplitter(chunk_size=chunk_size, chunk_overlap=chunk_overlap)
        documents = text_splitter.split_documents(documents)
        fixed_documents = []
        for doc in documents:
            doc.page_content = self.process_text(doc.page_content)
            if not doc.page_content:
                continue
            fixed_documents.append(doc)

        ids = []
        for path in file_paths:
            for i in range(1, len(fixed_documents) + 1):
                ids.append(f"{path.split('/')[-1].replace('.txt', '')}{i}")

        if db:
            data = db.get()
            files_db = {dict_data['source'].split('/')[-1] for dict_data in data["metadatas"]}
            files_load = {dict_data.metadata["source"].split('/')[-1] for dict_data in fixed_documents}
            if files_load == files_db:
                db.update_documents(ids, fixed_documents)
                return db, file_warning

        db = Chroma.from_documents(
            documents=fixed_documents,
            embedding=self.embeddings,
            ids=ids,
            client_settings=Settings(
                anonymized_telemetry=False,
                persist_directory="db"
            )
        )
        file_warning = f"Загружено {len(fixed_documents)} фрагментов! Можно задавать вопросы."

        return db, file_warning

    @staticmethod
    def user(message, history):
        new_history = history + [[message, None]]
        return "", new_history

    @staticmethod
    def regenerate_response(history):
        return "", history

    @staticmethod
    def retrieve(history, db, retrieved_docs):
        if db:
            last_user_message = history[-1][0]
            docs = db.similarity_search(last_user_message)
            source_docs = set()
            for doc in docs:
                for content in doc.metadata.values():
                    source_docs.add(content.split("/")[-1])
            retrieved_docs = "\n\n".join([doc.page_content for doc in docs])
            retrieved_docs = f"Документ - {''.join(list(source_docs))}.\n\n{retrieved_docs}"
        return retrieved_docs

    def bot(self, history, retrieved_docs, top_p, top_k, temp, model_selector):
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

    def run(self):
        with gr.Blocks(theme=gr.themes.Soft()) as demo:
            db = gr.State(None)
            favicon = f'<img src="{FAVICON_PATH}" width="48px" style="display: inline">'
            gr.Markdown(
                f"""<h1><center>{favicon} Я Лисум, текстовый ассистент на основе GPT</center></h1>"""
            )

            with gr.Accordion("Параметры", open=False):
                with gr.Tab(label="Параметры извлечения фрагментов из текста"):
                    gr.Slider(
                        minimum=1,
                        maximum=10,
                        value=2,
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
                with gr.Column(scale=3):
                    file_output = gr.Files(file_count="multiple", label="Загрузка файлов")
                    file_paths = gr.State([])
                    file_warning = gr.Markdown("Фрагменты ещё не загружены!")

            with gr.Row(elem_id="model_selector_row"):
                model_selector = gr.Dropdown(
                    choices=MODELS,
                    value=MODELS[0] if len(MODELS) > 0 else "",
                    interactive=True,
                    show_label=False,
                    container=False,
                )

            with gr.Row():
                with gr.Column(scale=5):
                    chatbot = gr.Chatbot(label="Диалог", height=400)
                with gr.Column(min_width=200, scale=4):
                    retrieved_docs = gr.Textbox(
                        label="Извлеченные фрагменты",
                        placeholder="Появятся после задавания вопросов",
                        interactive=False,
                        height=400
                    )

            with gr.Row():
                with gr.Column(scale=20):
                    msg = gr.Textbox(
                        label="Отправить сообщение",
                        show_label=False,
                        placeholder="Отправить сообщение",
                        container=False
                    )
                with gr.Column(scale=3, min_width=100):
                    submit = gr.Button("📤 Отправить")

            with gr.Row():
                gr.Button(value="👍  Понравилось")
                gr.Button(value="👎  Не понравилось")
                stop = gr.Button(value="⛔ Остановить")
                regenerate = gr.Button(value="🔄  Повторить")
                clear = gr.Button(value="🗑️  Очистить")

            # Upload files
            file_output.upload(
                fn=self.upload_files,
                inputs=[file_output],
                outputs=[file_paths],
                queue=True,
            ).success(
                fn=self.build_index,
                inputs=[file_paths, db, chunk_size, chunk_overlap, file_warning],
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
                inputs=[chatbot, db, retrieved_docs],
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
                inputs=[chatbot, db, retrieved_docs],
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
                inputs=[chatbot, db, retrieved_docs],
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

        demo.queue(max_size=128, concurrency_count=1)
        demo.launch()


if __name__ == "__main__":
    local_chat_gpt = LocalChatGPT()
    local_chat_gpt.run()
