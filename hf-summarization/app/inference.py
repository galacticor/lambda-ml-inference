from transformers import T5Tokenizer, T5ForConditionalGeneration

_tokenizer = None
_model = None


def predict(text: str) -> str:
    tokenizer = get_tokenizer()
    input_ids = tokenizer.encode(text, return_tensors='pt')

    model = get_model()
    summary_ids = model.generate(input_ids,
                max_length=100, 
                num_beams=2,
                repetition_penalty=2.5, 
                length_penalty=1.0, 
                early_stopping=True,
                no_repeat_ngram_size=2,
                use_cache=True)

    return tokenizer.decode(summary_ids[0], skip_special_tokens=True)



def get_tokenizer():
    global _tokenizer

    if _tokenizer is not None:
        return _tokenizer

    print("Loading the tokenizer")
    _tokenizer = T5Tokenizer.from_pretrained("./pipeline/")
    print("Loading tokenizer is successful")
    return _tokenizer


def get_model():
    global _model

    if _model is not None:
        return _model

    print("Loading the model")
    _model = T5ForConditionalGeneration.from_pretrained("./model")
    print("Loading model is successful")
    return _model