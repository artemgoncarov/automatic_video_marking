from llama_cpp import Llama


model_path = 'model-q2_K.gguf'
model_saiga = Llama(
    model_path=model_path,
    n_ctx=8192,
    n_parts=1,
    verbose=False,
)

def interact(
        messages,
        top_k=30,
        top_p=0.9,
        temperature=0.6,
        repeat_penalty=1.1,
):
    s = ''
    for part in model_saiga.create_chat_completion(
            messages,
            temperature=temperature,
            top_k=top_k,
            top_p=top_p,
            repeat_penalty=repeat_penalty,
            stream=True,
    ):
        delta = part["choices"][0]["delta"]
        if "content" in delta:
            print(delta["content"], end="", flush=True)
            s += delta["content"]

    return s