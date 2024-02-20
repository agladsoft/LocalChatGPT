import re
import logging
from __init__ import *
from datetime import datetime
from template import create_doc
from logging_custom import FileLogger
from app import app_celery, LocalChatGPT, LINEBREAK_TOKEN, MODES, BOT_TOKEN, MAX_NEW_TOKENS, SOURCES_SEPARATOR


logger = logging.getLogger(__name__)
f_logger = FileLogger(__name__, f"{LOGGING_DIR}/answers_bot.log", mode='a', level=logging.INFO)


class LLM(LocalChatGPT):
    def __init__(self):
        super().__init__()
        self.llama_models = self.initialize_app()
    
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

    def bot(self, history, collection_radio, retrieved_docs, top_p, top_k, temp, model_selector, scores, message):
        """

        :param history:
        :param collection_radio:
        :param retrieved_docs:
        :param top_p:
        :param top_k:
        :param temp:
        :param model_selector:
        :param scores:
        :param message:
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


llm_model = None


@app_celery.task(bind=True)
def receive_answer(self, chatbot, collection_radio, retrieved_docs, top_p, top_k, temp, model_selector, scores,
                   message=""):

    global llm_model

    if not llm_model:
        llm_model = LLM()
    letters = None
    chatbot = llm_model.bot(chatbot, collection_radio, retrieved_docs, top_p, top_k, temp, model_selector, scores,
                            message=message)
    for letters in chatbot:
        self.update_state(state='PROGRESS', meta={'progress': letters})
    self.update_state(state='SUCCESS', meta={'result': letters})
    return letters[-1][1]
