local CURRENT_DIR = std.extVar("CURRENT_DIR");
local DATA_CONFIG_DIR = "%s/data_configs" % CURRENT_DIR;
local ROOT_DIR = std.extVar("ROOT");
local DATSET_DIR = "%s/datasets" % ROOT_DIR;

local combined_records_confog = import "./combined_all_record_config.jsonnet";


{
  util_name: "DatasetSpliter",
  source_data_file_path: combined_records_confog.save_file_path,
  save_file_paths: [
    "%s/record_fro_%s_%s.pkl" % [DATSET_DIR, "train", std.extVar("TAG")],
    "%s/record_fro_%s_%s.pkl" % [DATSET_DIR, "valid", std.extVar("TAG")],
    "%s/record_fro_%s_%s.pkl" % [DATSET_DIR, "test", std.extVar("TAG")],
  ],
  ratios: [80, 10, 10],
}
