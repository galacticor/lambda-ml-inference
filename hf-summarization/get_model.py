from transformers import T5ForConditionalGeneration

model = T5ForConditionalGeneration.from_pretrained("panggi/t5-small-indonesian-summarization-cased")
model.save_pretrained('./model')
