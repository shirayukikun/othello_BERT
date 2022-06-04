LIB_DIR="${HOME}/lab/private/othello_BERT/lib"
TRANSFORMERS_DIR="${LIB_DIR}/transformers"
LIBEDAX_DIR="${LIB_DIR}/libedax4py-0.1.1"
OTHELLOLIB_DIR="${LIB_DIR}/othello_lib"
DOCKER_SETTING_DIR="${HOME}/lab/private/othello_BERT/docker_setting"


cd ${DOCKER_SETTING_DIR}
wandb login `cat .wandb_api_key.txt`
cd

cd ${TRANSFORMERS_DIR}
echo 'Installing transformers...'
pip install --editable .
cd

cd ${LIBEDAX_DIR}
echo 'Installing requirements...'
pip install --editable .
cd

cd ${OTHELLOLIB_DIR}
echo 'Installing requirements...'
pip install --editable .
cd


ENTER_DIR=`cat ${DOCKER_SETTING_DIR}/enter_dir.txt`
cd $ENTER_DIR

echo 'done.'
zsh
