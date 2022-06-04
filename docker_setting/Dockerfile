FROM nvcr.io/nvidia/pytorch:21.06-py3

ENV HOME="/home/keitonlp" DEBIAN_FRONTEND="noninteractive" TZ="Asia/Tokyo" 




RUN mkdir -p ~/.zsh/ && \
    cd ~/.zsh && \
    git clone https://github.com/zsh-users/zsh-autosuggestions.git

RUN apt update && apt install -y \
    zsh \
    tzdata \
    jsonnet



COPY requirements.txt /setup_dir/requirements.txt
RUN pip install -r /setup_dir/requirements.txt

RUN chmod -R 777 ${HOME}

COPY load_entrypoint.sh /setup_dir/load_entrypoint.sh
ENTRYPOINT ["zsh", "/setup_dir/load_entrypoint.sh"]

