import re
from __init__ import *
from datetime import datetime
from logging_custom import FileLogger
from app import LocalChatGPT, LINEBREAK_TOKEN, MODES, BOT_TOKEN, MAX_NEW_TOKENS, SOURCES_SEPARATOR


logger = logging.getLogger(__name__)

if not os.path.exists(LOGGING_DIR):
    os.mkdir(LOGGING_DIR)
f_logger = FileLogger(__name__, f"{LOGGING_DIR}/answers_bot.log", mode='a', level=logging.INFO)


class LLM(LocalChatGPT):
    def __init__(self):
        super().__init__()
        self.llama_models, self.embeddings = self.initialize_app()

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
        if retrieved_docs and collection_radio == MODES[0]:
            last_user_message = f"Контекст: {retrieved_docs}\n\nИспользуя только контекст, ответь на вопрос: " \
                                    f"{last_user_message}"
        message_tokens = self.get_message_tokens(model=model, role="user", content=last_user_message)
        tokens.extend(message_tokens)
        logger.info("Вопрос был полностью сформирован")
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
        # logger.info("Осуществляется генерации ответа")
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
        if files:
            partial_text += SOURCES_SEPARATOR
            sources_text = "\n\n\n".join(
                f"{index}. {source}"
                for index, source in enumerate(files, start=1)
            )
            partial_text += sources_text
            if scores and scores[0] > 4:
                partial_text += f"\n\n⚠️ Похоже, данные в Базе знаний слабо соответствуют вашему запросу. " \
                                f"Попробуйте подробнее описать ваш запрос или перейти в режим {MODES[1]}, " \
                                f"чтобы общаться с Макаром вне контекста Базы знаний"
            history[-1][1] = partial_text
        yield history


if __name__ == "__main__":
    local_chat_gpt = LocalChatGPT()
    local_chat_gpt.run()
