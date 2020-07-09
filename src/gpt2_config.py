from transformers import GPT2Tokenizer, GPT2LMHeadModel

GPT2_TOKENIZER = GPT2Tokenizer.from_pretrained('gpt2-medium')
GPT2_MODEL = GPT2LMHeadModel.from_pretrained('gpt2-medium')

