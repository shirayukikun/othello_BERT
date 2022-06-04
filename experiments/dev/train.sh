set -eu
source ./setup.sh

STDOUT_DIR="${ROOT_DIR}/stdout_logs"
GENERATED_JSON_CONFIG="./configs/_train_config_${TAG}.json"


if [ $# = 2 ]; then
    config_file_path=$1
    gpu_id=$2
else
    echo "Select config file path and gpu id"
    exit
fi


jsonnet $config_file_path --ext-str TAG=${TAG} --ext-str ROOT=${ROOT_DIR} --ext-str CURRENT_DIR=${CURRENT_DIR} > $GENERATED_JSON_CONFIG

echo "Config file :"
cat $GENERATED_JSON_CONFIG
TRAIN_ARGS=`python ./tools/config2args.py ${GENERATED_JSON_CONFIG}`

echo $TRAIN_ARGS

export CUDA_VISIBLE_DEVICES=`python -c "import torch;print(\",\".join(map(str, range(torch.cuda.device_count()))))"`

cd ${SOURCE_DIR}
TRAIN_PROGRAM=`echo "python ./train.py --mode train --gpu_id ${gpu_id} ${TRAIN_ARGS} 2>&1 | tee ${STDOUT_DIR}/train_${TAG}_${DATE}.log"`
eval $TRAIN_PROGRAM
cd -

python ./tools/train_fin_nortification/nortificate_program_fin.py -m "TAG = ${TAG}, Train finish!!"
