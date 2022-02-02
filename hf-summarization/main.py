from app.inference import predict
from aws_lambda_powertools import Logger, Tracer
from aws_lambda_powertools.logging import correlation_paths
from aws_lambda_powertools.event_handler.api_gateway import ApiGatewayResolver, ProxyEventType, CORSConfig

tracer = Tracer()
logger = Logger()
cors_config = CORSConfig(allow_origin="*", max_age=300)
app = ApiGatewayResolver(
    proxy_type=ProxyEventType.APIGatewayProxyEventV2,
    cors=cors_config
)


@app.get("/hello")
@tracer.capture_method
def get_hello_universe():
    return {"message": "hello universe"}


@logger.inject_lambda_context(correlation_id_path=correlation_paths.API_GATEWAY_HTTP)
@tracer.capture_lambda_handler
def lambda_handler(event, context):
    return app.resolve(event, context)


@app.post("/summarize")
@tracer.capture_method
def summarize():
    json_payload = app.current_event.json_body
    text = json_payload.get('text')

    if not text:
        return {}

    summarize_text = predict(text)
    print(summarize_text)

    return {'result': summarize_text}
