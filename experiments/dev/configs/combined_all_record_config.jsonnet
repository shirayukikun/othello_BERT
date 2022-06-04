local CURRENT_DIR = std.extVar("CURRENT_DIR");
local DATA_CONFIG_DIR = "%s/data_configs" % CURRENT_DIR;
local ROOT_DIR = std.extVar("ROOT");
local DATSET_DIR = "%s/datasets" % ROOT_DIR;

local source_configs = [
  import "./generate_black_slop_white_best.jsonnet",
  import "./generate_black_best_white_slop.jsonnet",
  import "./generate_black_random_white_best.jsonnet",
  import "./generate_black_best_white_random.jsonnet",
];


{
  util_name: "DatasetCombineder",
  save_file_path: "%s/combined_records_%s.pkl" % [
    DATSET_DIR,
    std.extVar("TAG"),
  ],
  source_data_file_paths: [
    config.save_file_path for config in source_configs
  ],
  shuffle: true,
}
