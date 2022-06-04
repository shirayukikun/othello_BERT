set -eu
source ./setup.sh

if [ $# = 1 ]; then
    config_file_path=$1
else
    echo "select config file path"
    exit
fi

output_josn_path="./configs/_intermediate_use_util_config.json"
jsonnet $config_file_path --ext-str TAG=${TAG} --ext-str ROOT=${ROOT_DIR} --ext-str CURRENT_DIR=${CURRENT_DIR} > $output_josn_path


echo "Config file :"
cat $output_josn_path
PREPRO_ARGS=`python ./tools/config2args.py ${output_josn_path}`

echo $PREPRO_ARGS

cd ${SOURCE_DIR}
PREPRO_PROGRAM=`echo "python ./use_utils.py ${PREPRO_ARGS}"`
eval $PREPRO_PROGRAM
cd -

python ./tools/train_fin_nortification/nortificate_program_fin.py -m "TAG = ${TAG}, Utility process was finished!!"
