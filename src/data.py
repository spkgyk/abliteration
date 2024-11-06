from datasets import load_dataset, DatasetDict


def reformat_texts(texts):
    """
    Reformats a list of text strings into a list of dictionaries with user role and content.

    Args:
        texts (List[str]): List of text strings to be reformatted.

    Returns:
        List[List[Dict]]: List of lists containing dictionaries with 'role' and 'content' keys.
        Each inner list contains a single dictionary representing a user message.
    """
    return [[{"role": "user", "content": text}] for text in texts]


class HarmfulHarmlessData:
    """
    A class for loading and managing datasets of harmful and harmless instructions.

    This class provides functionality to load and process two types of datasets:
    1. Harmful instructions from 'mlabonne/harmful_behaviors'
    2. Harmless instructions from 'mlabonne/harmless_alpaca'

    The datasets are loaded with specified sizes for training and testing sets,
    and reformatted into a consistent structure for use in training models.

    Attributes:
        n_inst_train (int): Number of instances to include in training sets.
        n_inst_test (int): Number of instances to include in test sets.
        harmful (DatasetDict): Dataset containing harmful instructions.
        harmless (DatasetDict): Dataset containing harmless instructions.
    """

    def __init__(self, n_inst_train: int = 2048, n_inst_test: int = 16) -> None:
        """
        Initialize the HarmfulHarmlessData instance.

        Args:
            n_inst_train (int): Number of instances to include in training sets. Defaults to 2048.
            n_inst_test (int): Number of instances to include in test sets. Defaults to 16.
        """
        self.n_inst_train = n_inst_train
        self.n_inst_test = n_inst_test
        self.harmful = None
        self.harmless = None

    def _get_harmful_instructions(self):
        """
        Load and process the harmful instructions dataset.

        Loads the 'mlabonne/harmful_behaviors' dataset and reformats it into
        training and test sets of specified sizes.

        Returns:
            DatasetDict: Processed dataset containing harmful instructions split into
                        train and test sets.
        """
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
        """
        Load and process the harmless instructions dataset.

        Loads the 'mlabonne/harmless_alpaca' dataset and reformats it into
        training and test sets of specified sizes.

        Returns:
            DatasetDict: Processed dataset containing harmless instructions split into
                        train and test sets.
        """
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
        """
        Load both harmful and harmless instruction datasets.

        This method loads and processes both types of datasets, storing them
        in the instance attributes 'harmful' and 'harmless'.
        """
        self.harmful = self._get_harmful_instructions()
        self.harmless = self._get_harmless_instructions()
