from transformers import BertTokenizer, BertForMaskedLM

BERT_TOKENIZER = BertTokenizer.from_pretrained('bert-large-uncased')
BERT_MODEL = BertForMaskedLM.from_pretrained('bert-large-uncased')

