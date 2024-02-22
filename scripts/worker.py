from __init__ import *
from celery import Celery


celery = Celery(__name__)
celery.conf.broker_url = os.environ.get("CELERY_BROKER_URL", "redis://localhost:6379")
celery.conf.result_backend = os.environ.get("CELERY_RESULT_BACKEND", "redis://localhost:6379")


llm_model = None


@celery.task(name="create_task", bind=True)
def receive_answer(self, chatbot, collection_radio, retrieved_docs, top_p, top_k, temp, model_selector, scores,
                   message=""):
    from llm import LLM

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
