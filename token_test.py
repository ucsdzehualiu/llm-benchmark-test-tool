from transformers import AutoTokenizer

tokenizer = AutoTokenizer.from_pretrained("Qwen/QwQ-32B")
tokens = tokenizer.encode("Your text here")