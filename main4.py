import gradio as gr

import copy
import random
import os
import requests
import time
import sys

from huggingface_hub import snapshot_download
from llama_cpp import Llama


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


def get_message_tokens(model, role, content):
    message_tokens = model.tokenize(content.encode("utf-8"))
    message_tokens.insert(1, ROLE_TOKENS[role])
    message_tokens.insert(2, LINEBREAK_TOKEN)
    message_tokens.append(model.token_eos())
    return message_tokens


def get_system_tokens(model):
    system_message = {"role": "system", "content": SYSTEM_PROMPT}
    return get_message_tokens(model, **system_message)


repo_name = "IlyaGusev/saiga2_13b_gguf"
model_name = "model-q4_K.gguf"

snapshot_download(repo_id=repo_name, local_dir=".", allow_patterns=model_name)

model = Llama(
    model_path=model_name,
    n_ctx=2000,
    n_parts=1,
)

max_new_tokens = 1500

def user(message, history):
    new_history = history + [[message, None]]
    return "", new_history


def bot(
    history,
    system_prompt,
    top_p,
    top_k,
    temp
):
    tokens = get_system_tokens(model)[:]
    tokens.append(LINEBREAK_TOKEN)

    for user_message, bot_message in history[:-1]:
        message_tokens = get_message_tokens(model=model, role="user", content=user_message)
        tokens.extend(message_tokens)
        if bot_message:
            message_tokens = get_message_tokens(model=model, role="bot", content=bot_message)
            tokens.extend(message_tokens)

    last_user_message = history[-1][0]
    message_tokens = get_message_tokens(model=model, role="user", content=last_user_message)
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
        if token == model.token_eos() or (max_new_tokens is not None and i >= max_new_tokens):
            break
        partial_text += model.detokenize([token]).decode("utf-8", "ignore")
        history[-1][1] = partial_text
        yield history


with gr.Blocks(
    theme=gr.themes.Soft()
) as demo:
    favicon = '<img src="https://cdn.midjourney.com/b88e5beb-6324-4820-8504-a1a37a9ba36d/0_1.png" width="48px" style="display: inline">'
    gr.Markdown(
        f"""<h1><center>{favicon}Saiga2 13B GGUF Q4_K</center></h1>
        This is a demo of a **Russian**-speaking LLaMA2-based model. If you are interested in other languages, please check other models, such as [MPT-7B-Chat](https://huggingface.co/spaces/mosaicml/mpt-7b-chat).
        Это демонстрационная версия [квантованной Сайги-2 с 13 миллиардами параметров](https://huggingface.co/IlyaGusev/saiga2_13b_ggml), работающая на CPU.
        Сайга-2 — это разговорная языковая модель, которая основана на [LLaMA-2](https://ai.meta.com/llama/) и дообучена на корпусах, сгенерированных ChatGPT, таких как [ru_turbo_alpaca](https://huggingface.co/datasets/IlyaGusev/ru_turbo_alpaca), [ru_turbo_saiga](https://huggingface.co/datasets/IlyaGusev/ru_turbo_saiga) и [gpt_roleplay_realm](https://huggingface.co/datasets/IlyaGusev/gpt_roleplay_realm).
        """
    )
    with gr.Row():
        with gr.Column(scale=5):
            system_prompt = gr.Textbox(label="Системный промпт", placeholder="", value=SYSTEM_PROMPT, interactive=False)
            chatbot = gr.Chatbot(label="Диалог").style(height=400)
        with gr.Column(min_width=80, scale=1):
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
                    value=0.01,
                    step=0.01,
                    interactive=True,
                    label="Температура"
                )
    with gr.Row():
        with gr.Column():
            msg = gr.Textbox(
                label="Отправить сообщение",
                placeholder="Отправить сообщение",
                show_label=False,
            ).style(container=False)
        with gr.Column():
            with gr.Row():
                submit = gr.Button("Отправить")
                stop = gr.Button("Остановить")
                clear = gr.Button("Очистить")
    with gr.Row():
        gr.Markdown(
            """ПРЕДУПРЕЖДЕНИЕ: Модель может генерировать фактически или этически некорректные тексты. Мы не несём за это ответственность."""
        )

    # Pressing Enter
    submit_event = msg.submit(
        fn=user,
        inputs=[msg, chatbot],
        outputs=[msg, chatbot],
        queue=False,
    ).success(
        fn=bot,
        inputs=[
            chatbot,
            system_prompt,
            top_p,
            top_k,
            temp
        ],
        outputs=chatbot,
        queue=True,
    )

    # Pressing the button
    submit_click_event = submit.click(
        fn=user,
        inputs=[msg, chatbot],
        outputs=[msg, chatbot],
        queue=False,
    ).success(
        fn=bot,
        inputs=[
            chatbot,
            system_prompt,
            top_p,
            top_k,
            temp
        ],
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

    # Clear history
    clear.click(lambda: None, None, chatbot, queue=False)

demo.queue(max_size=128, concurrency_count=1)
demo.launch()
