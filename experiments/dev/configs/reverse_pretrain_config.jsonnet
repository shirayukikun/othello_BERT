local CONFIG_DIR = "./data_configs";
local ROOT_DIR = std.extVar("ROOT");
local DATSET_DIR = "%s/datasets" % ROOT_DIR;
local LEARNING_TYPE = "no_meta_learning";

local train_data_config = import "./prepro_config_train.jsonnet";
local valid_data_config = import "./prepro_config_valid.jsonnet";
local test_data_config = import "./prepro_config_test.jsonnet";


{
  global_setting: {
    pl_model_name: "OthelloBertMaskedLM",
    seed: 42,
    tag: "%s_reverse_pretrain" % std.extVar("TAG"),
    log_model_output_dir: "%s/experiment_results/%s/%s" % [ROOT_DIR, self.tag, LEARNING_TYPE],
  },

  Logger: {
    project_name: "othello_BERT",
    log_dir: "%s/logs" % $.global_setting.log_model_output_dir,
    version: "%s/%s/train" % [$.global_setting.tag, LEARNING_TYPE],
  },

  
  Trainer: {
    max_epochs: 20,
    val_check_interval: 1000,
    check_val_every_n_epoch: 1,
    default_root_dir: "%s/defaults" % $.global_setting.log_model_output_dir,
    weights_save_path: "%s/weights" % $.global_setting.log_model_output_dir,
    fp16: false,
    #fast_dev_run: 10,
  },
  
  Callbacks: {
    save_top_k: 3,
    checkpoint_save_path: "%s/checkpoints" % $.global_setting.log_model_output_dir,
    early_stopping_patience: 5,
  },

  
  pl_module_setting: {
    lr: 1e-6,
    end_lr: self.lr * 0.1,
    warmup_steps_ratio: 0.0,
    power: 0.5,
    
    attention_probs_dropout_prob: 0.1,
    hidden_dropout_prob: 0.1,
    num_hidden_layers: 16,
    num_attention_heads: 16,
    hidden_size: 768,
    
    model_name_or_path: "%s/model_config/OthelloBert.config" % ROOT_DIR,
    from_scratch: true,
    
  },

  Datasets: {
    train_data_file_path: train_data_config.save_file_path,
    valid_data_file_path: valid_data_config.save_file_path,
    test_data_file_paths: [
      test_data_config.save_file_path,
    ],
    batch_size: 64,
    num_workers: 4,
    train_data_shuffle: true,
  },
	     
}
