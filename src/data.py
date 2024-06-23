from datasets import load_dataset


def reformat_texts(texts):
    return [[{"role": "user", "content": text}] for text in texts[:1000]]


def get_harmful_instructions():
    dataset = load_dataset("mlabonne/harmful_behaviors")
    return reformat_texts(dataset["train"]["text"]), reformat_texts(dataset["test"]["text"])


def get_harmless_instructions():
    dataset = load_dataset("mlabonne/harmless_alpaca")
    return reformat_texts(dataset["train"]["text"]), reformat_texts(dataset["test"]["text"])
