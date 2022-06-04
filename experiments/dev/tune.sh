set -eu
source ./setup.sh

STDOUT_DIR="${ROOT_DIR}/stdout_logs"
GENERATED_JSON_CONFIG="./configs/_tune_config_${TAG}.json"


if [ $# = 1 ]; then
    config_file_path=$1
else
    echo "Select train config file path"
    exit
fi


jsonnet $config_file_path --ext-str TAG=${TAG} --ext-str ROOT=${ROOT_DIR} --ext-str CURRENT_DIR=${CURRENT_DIR} > $GENERATED_JSON_CONFIG

echo "Config file :"
cat $GENERATED_JSON_CONFIG
TUNE_ARGS=`python ./tools/config2args.py ${GENERATED_JSON_CONFIG}`

echo $TUNE_ARGS

export CUDA_VISIBLE_DEVICES=`python -c "import torch;print(\",\".join(map(str, range(torch.cuda.device_count()))))"`

cd ${SOURCE_DIR}
TUNE_PROGRAM=`echo "python ./tune.py ${TUNE_ARGS} 2>&1 | tee ${STDOUT_DIR}/tune_${TAG}_${DATE}.log"`
eval $TUNE_PROGRAM
cd -

python ./tools/train_fin_nortification/nortificate_program_fin.py -m "TAG = ${TAG}, Tunning finish!!"
