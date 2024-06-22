from transformers import AutoTokenizer

# Load the tokenizer (replace 'bert-base-uncased' with the tokenizer you are using)
tokenizer = AutoTokenizer.from_pretrained("Qwen/Qwen1.5-4B-Chat")

# Words to tokenize
words = [
    " cannot",
    " sorry",
    "However",
    " unsafe",
    "danger",
    "'t",
]

# words = ["A", "To", "Here", "Certainly", "Absolutely"]

# Tokenize the words
token_ids = tokenizer(words, add_special_tokens=False)["input_ids"]

print([item for sublist in token_ids for item in sublist])

# Print the tokens and their corresponding IDs
for word, token_id in zip(words, token_ids):
    tokens = tokenizer.convert_ids_to_tokens(token_id)
    print(f"Word: {word}")
    print(f"Token IDs: {token_id}")
    print(f"Tokens: {tokens}")
    print("-" * 30)
