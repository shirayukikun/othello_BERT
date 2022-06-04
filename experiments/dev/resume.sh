set -eu
source ./setup.sh

STDOUT_DIR="${ROOT_DIR}/stdout_logs"
GENERATED_JSON_CONFIG="./configs/_resume_train_config_${TAG}.json"


if [ $# = 3 ]; then
    config_file_path=$1
    gpu_id=$2
    ckpt_file_path=$3
else
    echo "Select train config file path, gpu_id and check point path"
    exit
fi


jsonnet "./configs/resume_config.jsonnet" --ext-str TAG=${TAG} --ext-str ROOT=${ROOT_DIR} --ext-str CKPT_FILE_PATH=${ckpt_file_path}  --ext-code-file ORIGINAL_CONF=${config_file_path} --ext-str CURRENT_DIR=${CURRENT_DIR} > $GENERATED_JSON_CONFIG

echo "Config file :"
cat $GENERATED_JSON_CONFIG
TRAIN_ARGS=`python ./tools/config2args.py ${GENERATED_JSON_CONFIG}`

echo $TRAIN_ARGS

export CUDA_VISIBLE_DEVICES=`python -c "import torch;print(\",\".join(map(str, range(torch.cuda.device_count()))))"`

cd ${SOURCE_DIR}
TRAIN_PROGRAM=`echo "python ./train.py --mode resume --gpu_id ${gpu_id} ${TRAIN_ARGS} 2>&1 | tee ${STDOUT_DIR}/train_${TAG}_${DATE}.log"`
eval $TRAIN_PROGRAM
cd -

python ./tools/train_fin_nortification/nortificate_program_fin.py -m "TAG = ${TAG}, Train finish!!"
