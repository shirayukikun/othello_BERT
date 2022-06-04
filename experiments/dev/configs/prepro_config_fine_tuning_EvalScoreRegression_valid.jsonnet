local CURRENT_DIR = std.extVar("CURRENT_DIR");
local ROOT_DIR = std.extVar("ROOT");
local DATSET_DIR = "%s/datasets" % ROOT_DIR;
local record_config = import "./split_train_valid_test_fine_tuning.jsonnet";
local train_config = import "./prepro_config_fine_tuning_EvalScoreRegression_train.jsonnet";

{
  task: "EvalScoreRegrssionWithAttensionMask",
  save_file_path: "%s/%s_fine_tuning_%s_%s" % [
    DATSET_DIR,
    std.extVar("TAG"),
    self.task,
    "valid",
  ],
  record_file_path: record_config.save_file_paths[1], 
  exclude_dataset_paths: [
    train_config.save_file_path
  ],
  number_of_data: 1000,
}
