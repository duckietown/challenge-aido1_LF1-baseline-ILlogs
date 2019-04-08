FROM duckietown/aido2-base-python3:v4

COPY requirements.txt requirements.txt
RUN pip install -r requirements.txt
RUN pip list

COPY . .

ENV DISABLE_CONTRACTS=1
ENTRYPOINT ["python3", "scenario_maker.py"]
