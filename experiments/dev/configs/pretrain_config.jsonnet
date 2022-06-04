local CONFIG_DIR = "./data_configs";
local ROOT_DIR = std.extVar("ROOT");
local DATSET_DIR = "%s/datasets" % ROOT_DIR;
local LEARNING_TYPE = "pretrain";

local pretrain_data_config = import "./prepro_config_pretrain.jsonnet";
local valid_data_config = import "./prepro_config_valid.jsonnet";
local test_data_config = import "./prepro_config_test.jsonnet";



{
  global_setting: {
    pl_model_name: "DentakuBart",
    seed: 42,
    gpu_id: 0,
    use_gpu: true,
    tag: "%s" % std.extVar("TAG"),
    log_model_output_dir: "%s/experiment_results/%s/%s" % [ROOT_DIR, self.tag, LEARNING_TYPE],
  },

  TensorBoard: {
    log_dir: "%s/logs" % $.global_setting.log_model_output_dir,
    version: "%s/%s/train" % [$.global_setting.tag, LEARNING_TYPE],
  },


  Trainer: {
    max_epochs: 3,
    val_check_interval: 200,
    check_val_every_n_epoch: 1,
    default_root_dir: "%s/defaults" % $.global_setting.log_model_output_dir,
    weights_save_path: "%s/weights" % $.global_setting.log_model_output_dir,
  },
  
  Callbacks: {
    save_top_k: 3,
    checkpoint_save_path: "%s/checkpoints" % $.global_setting.log_model_output_dir,
  },

  
  pl_module_setting: {
    lr: 1e-5,
    end_lr: self.lr * 0.1,
    decay_steps: pretrain_data_config.max_steps * $.Trainer.max_epochs,
    num_warmup_steps: 0,
    power: 0.5,
    
    model_name_or_path: "facebook/bart-base",
    assert self.decay_steps + self.num_warmup_steps == pretrain_data_config.max_steps * $.Trainer.max_epochs,
    num_beams: 1,
    max_length: 10,
    from_scrach: true,
  },

  Datasets: {
    train_data_file_path: pretrain_data_config.save_file_path,
    valid_data_file_path: valid_data_config.save_file_path,
    test_data_file_path: test_data_config.save_file_path,
  },    
	     
}
