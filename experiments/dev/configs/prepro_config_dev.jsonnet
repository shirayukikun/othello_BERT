local CURRENT_DIR = std.extVar("CURRENT_DIR");
local ROOT_DIR = std.extVar("ROOT");
local DATSET_DIR = "%s/datasets" % ROOT_DIR;
local record_config = import "./generate_dev_config.jsonnet";

{
  task: "EvalScoreRegrssion",
  save_file_path: "%s/%s_%s_%s" % [
    DATSET_DIR,
    std.extVar("TAG"),
    self.task,
    "dev",
  ],
  record_file_path: record_config.save_file_path, 
  exclude_dataset_paths: [],
  number_of_data: 100,
}
