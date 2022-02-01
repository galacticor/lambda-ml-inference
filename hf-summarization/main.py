from transformers import T5Tokenizer, T5ForConditionalGeneration

tokenizer = T5Tokenizer.from_pretrained("./pipeline/")
model = T5ForConditionalGeneration.from_pretrained("panggi/t5-small-indonesian-summarization-cased")


def handler(event, context):
    print(event)
    text = event.get('text')

    input_ids = tokenizer.encode(text, return_tensors='pt')
    summary_ids = model.generate(input_ids,
                max_length=100, 
                num_beams=2,
                repetition_penalty=2.5, 
                length_penalty=1.0, 
                early_stopping=True,
                no_repeat_ngram_size=2,
                use_cache=True)
    summary_text = tokenizer.decode(summary_ids[0], skip_special_tokens=True)

    print(summary_text)
