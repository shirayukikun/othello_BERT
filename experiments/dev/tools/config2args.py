import json
import argparse
from pathlib import Path


def print_args(file_path):

    arg_texts = []
    with open(file_path, mode="r") as f:
        config_json_dict = json.load(f)
        config_json_list = []
        for key, value in config_json_dict.items():
            if type(value) is dict:
                for k, v in value.items():
                    config_json_list.append((k, v))
            else:
                config_json_list.append((key, value))
        
        
        for key, value in config_json_list:
            
            if type(value) is dict:
                raise NotImplementedError("Not support dict type config.")
            
            elif type(value) is list:
                assert all(map(lambda x: not ((type(x) is dict) or (type(x) is list)), value)), "Not support dict or lst type config in list."
                arg_texts.append(f"--{key} " + " ".join(map(lambda x: f"\"{str(x)}\"", value)))

            elif type(value) is bool:
                if value:
                    arg_texts.append(f"--{key}")
            else:
                arg_texts.append(f"--{key} \"{value}\"")
    
    #print(" \\\n".join(arg_texts))
    print(" ".join(arg_texts))
    

def main(args):
    print_args(args.config_filepath)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("config_filepath",  help="Select config file", type=Path)
    args = parser.parse_args()
    main(args)

