local ROOT_DIR = std.extVar("ROOT");
local DATSET_DIR = "%s/datasets" % ROOT_DIR;
local LEARNING_TYPE = "no_meta_learning";

local TRAIN_CONFIG = import "./basic_train_config_arith.jsonnet";
local train_data_config = import "./prepro_config_train.jsonnet";


TRAIN_CONFIG + {
  Logger: TRAIN_CONFIG.Logger + {
    version: "%s/%s/tune" % [$.global_setting.tag, LEARNING_TYPE],
    project_name: "tune_dentaku_pretrain_analyse",
  },						  

  
  Tune_configs: { 
    tune_method: "Optuna",
    tune_space_file_path: "%s/configs/tune_space.jsonnet" % std.extVar("CURRENT_DIR"),
  
    cpus_per_trial: 4,
    gpus_per_trial: 1,
    tune_num_samples: 60,
  }
}


#ASH setting
#tune_num_max_report:
# local num_report_per_epoch = std.floor(train_data_config.max_steps / $.Trainer.val_check_interval);
# local num_max_tune_epoch = std.floor($.Trainer.max_epochs / 5);
#num_report_per_epoch *  num_max_tune_epoch,
#(猶予期間)
#grace_period: std.floor(self.tune_num_max_report / 10),
#reduction_factor: 4,




#  Tune_configs: {
#    # Choice "PBT", "ASH"
#    tune_method: "PBT",
#    tune_space_file_path: "%s/configs/tune_space.jsonnet" % std.extVar("CURRENT_DIR"),
#    cpus_per_trial: 5,
#    gpus_per_trial: 1,
#    tune_num_samples: 16,

#    local max_steps =
#      std.floor(train_data_config.number_of_data / $.Datasets.batch_size) +
#      if train_data_config.number_of_data % $.Datasets.batch_size == 0 then 0 else 1,

#    # twice of validation step
#    perturbation_interval: std.floor(max_steps / $.Trainer.val_check_interval) * 2,
#    quantile_fraction: 0.25,
#    resample_probability: 0.25,
#  }
