set -eu

source ./prepro.sh ./configs/prepro_config_fine_tuning_EvalScoreRegression_train.jsonnet
source ./prepro.sh ./configs/prepro_config_fine_tuning_EvalScoreRegression_valid.jsonnet
source ./prepro.sh ./configs/prepro_config_fine_tuning_EvalScoreRegression_test.jsonnet

#source ./prepro.sh ./configs/prepro_config_EvalScoreRegression_valid.jsonnet
#source ./prepro.sh ./configs/prepro_config_EvalScoreRegression_test.jsonnet


#source ./prepro.sh ./configs/prepro_config_EvalScoreRegression_train.jsonnet
#source ./prepro.sh ./configs/prepro_config_EvalScoreRegression_valid.jsonnet
#source ./prepro.sh ./configs/prepro_config_EvalScoreRegression_test.jsonnet

#source ./prepro.sh ./configs/prepro_config_masked_train.jsonnet
#source ./prepro.sh ./configs/prepro_config_masked_valid.jsonnet
#source ./prepro.sh ./configs/prepro_config_masked_test.jsonnet


#source ./prepro.sh ./configs/prepro_config_train.jsonnet
#source ./prepro.sh ./configs/prepro_config_valid.jsonnet
#source ./prepro.sh ./configs/prepro_config_test.jsonnet

#source ./prepro.sh ./configs/prepro_config_zero_shot_test.jsonnet
#source ./prepro.sh ./configs/prepro_config_test_plus_1_step.jsonnet
#source ./prepro.sh ./configs/prepro_config_test_minus_1_step.jsonnet
