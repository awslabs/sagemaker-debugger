FROM python:3.7-slim-buster

RUN pip3 install sagemaker
RUN pip3 install scipy
RUN pip3 install papermill
RUN pip3 install jupyter==1.0.0
RUN pip3 install pandas
RUN pip3 install matplotlib

ENV PYTHONUNBUFFERED=TRUE

ADD rule_evaluation.py /
ADD /pip_packages /pip_packages
ADD profiler_report.ipynb /
ADD requirements.txt /

RUN pip3 install -r requirements.txt

ENTRYPOINT ["python3", "/rule_evaluation.py"]
