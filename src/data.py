from datasets import load_dataset, DatasetDict


def reformat_texts(texts):
    return [[{"role": "user", "content": text}] for text in texts]


class HarmfulHarmlessData:

    def __init__(self, n_inst_train: int = 2048, n_inst_test: int = 16) -> None:
        self.n_inst_train = n_inst_train
        self.n_inst_test = n_inst_test
        self.harmful = None
        self.harmless = None

    def _get_harmful_instructions(self):
        dataset = load_dataset("mlabonne/harmful_behaviors")
        dataset = DatasetDict(
            {
                "train": reformat_texts(dataset["train"]["text"][: self.n_inst_train]),
                "test": reformat_texts(dataset["test"]["text"][: self.n_inst_test]),
            }
        )
        self.n_inst_train = min(self.n_inst_train, len(dataset["train"]))
        self.n_inst_test = min(self.n_inst_test, len(dataset["test"]))
        return dataset

    def _get_harmless_instructions(self):
        dataset = load_dataset("mlabonne/harmless_alpaca")
        dataset = DatasetDict(
            {
                "train": reformat_texts(dataset["train"]["text"][: self.n_inst_train]),
                "test": reformat_texts(dataset["test"]["text"][: self.n_inst_test]),
            }
        )
        self.n_inst_train = min(self.n_inst_train, len(dataset["train"]))
        self.n_inst_test = min(self.n_inst_test, len(dataset["test"]))
        return dataset

    def load_data(self):
        self.harmful = self._get_harmful_instructions()
        self.harmless = self._get_harmless_instructions()
