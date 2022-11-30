import csv
import pandas as pd
import numpy as np

from transformers import AutoTokenizer, RobertaForCausalLM
from transformers import RobertaConfig
import torch

"""
A script for running a batch of paired sentences through a RoBERTa language model to extract sentence 
probabilities for the Grammatical Diversity CS 232 final project
Author: Skylar Kolisko
Date: 4/4/2022
"""
    
tokenizer = AutoTokenizer.from_pretrained("roberta-base")
config = RobertaConfig.from_pretrained("roberta-base")
config.is_decoder = True
model = RobertaForCausalLM.from_pretrained("roberta-base", config=config)

def get_sentence_probability(sent):
    """
    Feeds a single prompt into RoBERTa and returns its cross-entropy loss
    """
    inputs = tokenizer(sent, return_tensors="pt")
    outputs = model(**inputs,labels=inputs["input_ids"])
    loss = outputs.loss.detach().numpy()
    return(loss)

def main():
    # Prompt user for input file name
    input_file_name= input("Enter name of input sentences, with respect to current directory: ") #so_dont_I_sentprob.tsv
    sentence_probability_file = open(input_file_name)

    # make dataframe from input data
    input_data_df = pd.read_csv(sentence_probability_file, delimiter='\t', names=['Sentence_Number', 'Version_(A/B)','Task_Name', 'Data_Set', 'Sentence', 'A_score', 'B_score', 'Difference_A-B'])

    num_rows=len(input_data_df)
    # count no. of lines
    print("Number of lines present:",len(input_data_df))
    header = ['Sentence_Number', 'Version_(A/B)','Task_Name', 'Data_Set', 'Sentence', 'A_score', 'B_score', 'Difference_A-B'] #header consists label 'Sentence_Number', 'Version_(A/B)','Task_Name', 'Data_Set', 'Sentence'and add three fields. It will give the task score for the first sentence, the task score for the second sentence, and the difference between the two task scores.

    # iterate through sentence pairs
    for i in range(0,num_rows,2): #skip odd rows
        # Get sentence A loss
        sent_A = input_data_df.iat[i, 4]
        prob_A = get_sentence_probability(sent_A)
        print(f"Sentence {i+1}-A:",sent_A)
        print(f"Sentence {i+1}-A loss:",prob_A)

        # Get sentence B loss
        sent_B = input_data_df.iat[i+1, 4]
        prob_B = get_sentence_probability(sent_B)
        print(f"Sentence {i+1}-B:",sent_B)
        print(f"Sentence {i+1}-B loss:",prob_B)

        # Calculate sentence A/B difference
        dif_A_B = prob_A-prob_B
        print(f"Ex. {i+1} A/B difference in loss:",dif_A_B)

        input_data_df.iat[i, 5]=prob_A #input probA in even
        input_data_df.iat[i, 6]=prob_B #input probB in even
        input_data_df.iat[i, 7]=dif_A_B #input probB in even

        input_data_df.iat[i+1, 5]=prob_A #input probA in even
        input_data_df.iat[i+1, 6]=prob_B #input probB in even
        input_data_df.iat[i+1, 7]=dif_A_B #input probB in even
        
    print(input_data_df)
    
    #prompt user for output file name
    output_file_name = input("Enter name of empty output csv, with respect to current directory: ") #relatedness_output_example.csv
    input_data_df.to_csv(output_file_name, index=False)

main()

