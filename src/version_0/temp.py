import argparse
from pathlib import Path

from data_generator.numerical_data_generation import NumericDataGenarator
from data_generator.equation_editor import EquationEditor
from itertools import islice
from pprint import pprint


#from dentaku_tokenizer.tokenizer import BartDentakuTokenizer
#tokenizer = BartDentakuTokenizer.from_pretrained('facebook/bart-base')



def main(args):
    data_config_file_path = args.data_config_file_path
    edit_config_file_path = args.edit_config_file_path

    question_generator = NumericDataGenarator(data_config_file_path)
    #editor = EquationEditor(edit_config_file_path, question_generator)


    for operator_config, assignment_configs in islice(question_generator(generate_config=True), 10):

        """
        edit_func = editor.select_function()

        edited_operator_config, edited_assignment_configs = edit_func(
            operator_config,
            assignment_configs
        )

        pprint(edited_assignment_configs)
        """

        pqa = question_generator.get_pqa_triple_from_configs(
            operator_config,
            assignment_configs,
            separate=True
        ) 

        #answers.append(pqa[0][2])

        """
        edited_pqa_triple_list = question_generator.get_pqa_triple_from_configs(
            edited_operator_config,
            edited_assignment_configs,
            separate=True
        )
        """

        passage = pqa[0][0]
        question = pqa[0][1]
        answer = pqa[0][2]

        """
        tokenized_input = tokenizer([passage], [question], padding=True, truncation=True, return_tensors='pt')
        decode_result = []
        for tok_id in tokenized_input["input_ids"][0]:
            decode_result.append(tokenizer.decode([tok_id], skip_special_tokens=False, clean_up_tokenization_spaces=False))
        """

        print(pqa)
        #print(tokenized_input["input_ids"][0])
        #print(edited_pqa_triple_list)
        #print(decode_result)
        #print(edited_pqa_triple_list)
        print("\n\n")




    
if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("data_config_file_path", help="Select ckpt file path", type=Path)
    parser.add_argument("edit_config_file_path", help="Select ckpt file path", type=Path)
    args = parser.parse_args()
    main(args)
