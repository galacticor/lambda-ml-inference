from app.inference import predict


def handler(event, context):
    print(event)
    text = event.get('text')

    summarize_text = predict(text)
    print(summarize_text)
