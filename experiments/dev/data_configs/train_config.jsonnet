local SPLITED_FILE_PATH = std.split(std.thisFile, "/");
local THIS_FILE_NAME = SPLITED_FILE_PATH[std.length(SPLITED_FILE_PATH) - 1];
local DATA_TYPE = std.split(THIS_FILE_NAME, "_")[0];
local TAG = std.extVar("TAG");
local utils = import "./utils.jsonnet";
local NUM_SUBSTITUTE = 3;


{
  seed : utils.datatype_to_seed(DATA_TYPE),
  number_of_symbols : 26,
  max_number_of_question : "inf",

  number_file_path: "%s/datasets/number_datasets/%s_numbers.pkl" % [
    std.extVar("ROOT"),
    DATA_TYPE
  ],
  
  dtype : "int",
  shuffle_order: true,
  output_type : "ask_last_question",
  
  generation_rules : [
    
    {
      comment: "%sつの代入" % std.toString(NUM_SUBSTITUTE),
      type : "template",
      selection_probability : 1.0,
      
      assignment_format : [
	
	{
	  type : "Substitution",
	  format : ["num"]
	}  for i in std.range(1, NUM_SUBSTITUTE)
	
      ],
      
      
      operator : {
	type : ["Check"],
	selection_probabilities : [1.0], 
	format : [[i] for i in std.range(0, NUM_SUBSTITUTE - 1)]
      }
    },

    {
      comment: "二項演算のみ",
      type : "template",
      selection_probability : 1.0,
      
      assignment_format : [
	{
	  type: ["Add", "Sub", "Min", "Max"],
	  format: [["num", "num"]]
	}
      ],

      operator : {
	type : ["Check"],
	selection_probabilities : [1.0], 
	format : [-1]
      }
    }
    
  ]
}
