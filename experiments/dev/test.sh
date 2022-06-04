set -eu
source ./setup.sh

STDOUT_DIR="${ROOT_DIR}/stdout_logs"
GENERATED_JSON_CONFIG="./configs/_test_config_${TAG}.json"


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
TEST_ARGS=`python ./tools/config2args.py ${GENERATED_JSON_CONFIG}`

echo $TEST_ARGS

export CUDA_VISIBLE_DEVICES=`python -c "import torch;print(\",\".join(map(str, range(torch.cuda.device_count()))))"`

cd ${SOURCE_DIR}
TEST_PROGRAM=`echo "python ./test.py --gpu_id ${gpu_id} ${TEST_ARGS} 2>&1 | tee ${STDOUT_DIR}/test_${TAG}_${DATE}.log"`
eval $TEST_PROGRAM
cd -

python ./tools/train_fin_nortification/nortificate_program_fin.py -m "TAG = ${TAG}, Test finish!!"
