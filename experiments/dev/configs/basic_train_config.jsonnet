local CONFIG_DIR = "./data_configs";
local ROOT_DIR = std.extVar("ROOT");
local DATSET_DIR = "%s/datasets" % ROOT_DIR;
local LEARNING_TYPE = "no_meta_learning";

local train_data_config = import "./prepro_config_train.jsonnet";
local valid_data_config = import "./prepro_config_valid.jsonnet";
local test_data_config = import "./prepro_config_test.jsonnet";
local zero_shot_test_data_config = import "./prepro_config_zero_shot_test.jsonnet";


{
  global_setting: {
    pl_model_name: "DentakuBart",
    seed: 42,
    tag: "%s" % std.extVar("TAG"),
    log_model_output_dir: "%s/experiment_results/%s/%s" % [ROOT_DIR, self.tag, LEARNING_TYPE],
  },

  Logger: {
    project_name: "dentaku_compositional_ability_analyse",
    log_dir: "%s/logs" % $.global_setting.log_model_output_dir,
    version: "%s/%s/train" % [$.global_setting.tag, LEARNING_TYPE],
  },

  
  Trainer: {
    max_epochs: 2,
    val_check_interval: 1,
    check_val_every_n_epoch: 1,
    default_root_dir: "%s/defaults" % $.global_setting.log_model_output_dir,
    weights_save_path: "%s/weights" % $.global_setting.log_model_output_dir,
    fp16: false,
    #fast_dev_run: 10,
  },
  
  Callbacks: {
    save_top_k: 3,
    checkpoint_save_path: "%s/checkpoints" % $.global_setting.log_model_output_dir,
    early_stopping_patience: -1,
  },

  
  pl_module_setting: {
    local max_steps =
      std.floor(train_data_config.number_of_data / $.Datasets.batch_size) +
      if train_data_config.number_of_data % $.Datasets.batch_size == 0 then 0 else 1,
    lr: 1e-5,
    end_lr: self.lr * 0.1,
    decay_steps: max_steps * $.Trainer.max_epochs,
    num_warmup_steps: 0,
    power: 0.5,
    
    model_name_or_path: "facebook/bart-base",
    assert self.decay_steps + self.num_warmup_steps == max_steps * $.Trainer.max_epochs,
    from_scratch: false,
    num_beams: 1,
    max_length: 10,
    
    positional_encodeing_type: "learnable",
    encoder_shape: true,
    decoder_shape: false,
    max_noise_size: 10,
    
    encoder_layers: 6,
    decoder_layers: 6,

  },

  Datasets: {
    train_data_file_path: train_data_config.save_file_path,
    valid_data_file_path: valid_data_config.save_file_path,
    test_data_file_paths: [
      test_data_config.save_file_path,
      zero_shot_test_data_config.save_file_path,
    ],
    batch_size: 32,
    num_workers: 4,
    train_data_shuffle: true,
  },
	     
}
