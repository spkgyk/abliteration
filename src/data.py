from datasets import load_dataset


def reformat_texts(texts):
    return [[{"role": "user", "content": text}] for text in texts]


def get_harmful_instructions():
    dataset = load_dataset("mlabonne/harmful_behaviors")
    return {
        "train": reformat_texts(dataset["train"]["text"]),
        "test": reformat_texts(dataset["test"]["text"]),
    }


def get_harmless_instructions():
    dataset = load_dataset("mlabonne/harmless_alpaca")
    return {
        "train": reformat_texts(dataset["train"]["text"]),
        "test": reformat_texts(dataset["test"]["text"]),
    }
