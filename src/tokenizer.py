import tiktoken

def get_tokenizer():
    return tiktoken.get_encoding("gpt2")