local CURRENT_DIR = std.extVar("CURRENT_DIR");
local ROOT_DIR = std.extVar("ROOT");
local DATSET_DIR = "%s/datasets" % ROOT_DIR;
local record_config = import "./split_train_valid_test.jsonnet";

{
  task: "EvalScoreRegrssion",
  save_file_path: "%s/%s_%s_%s" % [
    DATSET_DIR,
    std.extVar("TAG"),
    self.task,
    "train",
  ],
  record_file_path: record_config.save_file_paths[0], 
  exclude_dataset_paths: [],
  number_of_data: 500000,
}
