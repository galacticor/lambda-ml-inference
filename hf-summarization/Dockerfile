FROM public.ecr.aws/lambda/python:3.8

COPY requirements.txt ./requirements.txt
RUN python3 -m pip install -r requirements.txt 

COPY ./get_model.py ./
RUN python get_model.py

COPY ./   ./

CMD ["main.handler"]
