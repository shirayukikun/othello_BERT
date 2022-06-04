local CURRENT_DIR = std.extVar("CURRENT_DIR");
local DATA_CONFIG_DIR = "%s/data_configs" % CURRENT_DIR;
local ROOT_DIR = std.extVar("ROOT");
local DATSET_DIR = "%s/datasets" % ROOT_DIR;


{
  dataset_name: "DentakuPreTrainDataset",
  data_tag: "%s" % std.extVar("TAG"),
  data_config_file_path: "%s/train_config.jsonnet" % DATA_CONFIG_DIR,
  batch_size: 32,
  max_steps: 100000 * 2,
  save_file_path: "%s/%s_%s_%s" % [
    DATSET_DIR,
    self.dataset_name,
    self.batch_size,
    self.data_tag
  ],
  exclude_datasets_paths: [],
}
