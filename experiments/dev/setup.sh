set -eu

export SOURCE_DIR="${HOME}/lab/private/othello_BERT/src/version_0"


if [ -e "/.dockerenv" ]; then
    echo "This environment is docker!"
    export ROOT_DIR="/work"
else
    Iam=`whoami`
    echo "This environment is Inui Lab server!"
    export ROOT_DIR="/work02/keitokudo/private/othello_BERT"
fi

export ROOT=$ROOT_DIR
mkdir -p "${ROOT_DIR}/stdout_logs"
mkdir -p "${ROOT_DIR}/datasets"


wandb login

#必ず設定する
export PYTHONHASHSEED=0

DATE=`date +%Y%m%d-%H%M%S`
export CURRENT_DIR=`pwd`
export TAG=`basename ${CURRENT_DIR}`


echo "Setup : ${TAG}"

### for debugging ###
date
uname -a
which python
python --version
