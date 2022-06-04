local CURRENT_DIR = std.extVar("CURRENT_DIR");
local DATA_CONFIG_DIR = "%s/data_configs" % CURRENT_DIR;
local ROOT_DIR = std.extVar("ROOT");
local DATSET_DIR = "%s/datasets" % ROOT_DIR;


{
  number_of_data: 10000,
  save_file_path: "%s/%s_%s_black_%s_white_%s_%s" % [
    DATSET_DIR,
    std.extVar("TAG"),
    self.number_of_data,
    self.black_selection_policy,
    self.white_selection_policy,
    self.level,
  ],
  level: 12,
  black_selection_policy: "best",
  white_selection_policy: "slop",
  seed: 42,
}
