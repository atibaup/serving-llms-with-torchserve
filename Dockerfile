# ARGs shared across stages
ARG ARCH="cpu"
ARG N_WORKERS=1
ARG HOME=/home/model-server
ARG MODEL_STORE_PATH=$HOME/model-store

FROM pytorch/torchserve:latest-cpu as model_archiver

# defaults will be taken from shared definitions above
ARG MODEL_NAME
ARG HOME
ARG MODEL_STORE_PATH

# copy custom handler and other dependencies (tokenizer config, serialized model)
COPY $MODEL_NAME/*.json $HOME/$MODEL_NAME/
COPY $MODEL_NAME/*.txt $HOME/$MODEL_NAME/
COPY handlers/$MODEL_NAME/custom_handler.py $HOME/$MODEL_NAME/custom_handler.py
COPY $MODEL_NAME/pytorch_model.bin $HOME/$MODEL_NAME/pytorch_model.bin

RUN ls -d $HOME/$MODEL_NAME/*.json > extra_files_list.txt
RUN if ls $HOME/$MODEL_NAME/*.txt &> /dev/null; then ls -d $HOME/$MODEL_NAME/*.txt >> extra_files_list.txt; fi

# create model archive file packaging model artifacts and dependencies
RUN torch-model-archiver -f \
  --model-name=$MODEL_NAME \
  --version=1.0 \
  --serialized-file=$HOME/$MODEL_NAME/pytorch_model.bin \
  --handler=$HOME/$MODEL_NAME/custom_handler.py \
  # reads file list from txt file and converts into comma-separated string
  --extra-files "$(echo $(tr '\n' ',' < extra_files_list.txt | sed 's/,$//'))" \
  --export-path=$MODEL_STORE_PATH


FROM pytorch/torchserve:latest-cpu as cpu

FROM pytorch/torchserve:latest-gpu as gpu

FROM ${ARCH} as service

# defaults will be taken from shared definitions at the top
ARG MODEL_NAME
ARG HOME
ARG MODEL_STORE_PATH
ARG N_WORKERS

# bring in artefact from previous stage
COPY --from=model_archiver $MODEL_STORE_PATH/* $MODEL_STORE_PATH/

# install dependencies that are not included in the base image
RUN pip install transformers accelerate

# create torchserve configuration file, for some reason default user
# doesn't have write permissions in $HOME
# Note the limitation of number of workers and the increase on the
# default worker timeout to account for the fact that LLMs
# need a lot of memory and are particularly slow when run on CPUs
USER root
RUN printf "service_envelope=json" > $HOME/config.properties && \
    printf "\ninference_address=http://0.0.0.0:7080" >> $HOME/config.properties && \
    printf "\nmanagement_address=http://0.0.0.0:7081" >> $HOME/config.properties && \
    printf "\ndefault_workers_per_model=$N_WORKERS" >> $HOME/config.properties && \
    printf "\ndefault_response_timeout=600" >> $HOME/config.properties
USER model-server

# expose health and prediction listener ports from the image
EXPOSE 7080
EXPOSE 7081

# set MODEL_NAME env for runtime use (CMD)
ENV MODEL_NAME=$MODEL_NAME
ENV HOME=$HOME
ENV MODEL_STORE_PATH=$MODEL_STORE_PATH


# run Torchserve HTTP serve to respond to prediction requests
CMD ["torchserve", \
     "--start", \
     "--ts-config=$HOME/config.properties", \
     "--models", \
     "$MODEL_NAME=$MODEL_NAME.mar", \
     "--model-store", \
     "$MODEL_STORE_PATH"]
