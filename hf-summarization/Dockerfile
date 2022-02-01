FROM public.ecr.aws/lambda/python:3.8

COPY requirements.txt ./requirements.txt

# RUN python3 -m pip install --upgrade pip
RUN python3 -m pip install wheel
RUN python3 -m pip install -r requirements.txt 

COPY ./   ./

CMD ["main.handler"]
