from transformers import AutoTokenizer

# Initialize the tokenizer
tokenizer = AutoTokenizer.from_pretrained("/home/spkgyk-linux/code/abliteration/meta-llama/Meta-Llama-3-8B-Instruct")

# List of numbers to tokenize
numbers_list1 = [4250, 14931, 89735, 20451, 11660, 11458, 956]
numbers_list2 = [32, 1271, 8586, 96556, 78145]


# Tokenize the strings
tokenized_output_1 = tokenizer.convert_ids_to_tokens(numbers_list1)
tokenized_output_2 = tokenizer.convert_ids_to_tokens(numbers_list2)

# Print the tokenized output
print(tokenized_output_1)

print()

print(tokenized_output_2)
