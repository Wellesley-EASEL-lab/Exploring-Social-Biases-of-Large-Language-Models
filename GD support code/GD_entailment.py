# cd Desktop:/proj232/232_final_project/task2/pt3_entailment
import csv
import pandas as pd
import numpy as np

from transformers import AutoTokenizer, AutoModelForSequenceClassification, RobertaConfig
import torch

"""
A script for running a batch of paired sentences through a RoBERTa natural language inference model 
for the Grammatical Diversity CS 232 final project
Author: Skylar Kolisko
Date: 4/24/2022
"""

tokenizer = AutoTokenizer.from_pretrained("roberta-large-mnli")
model = AutoModelForSequenceClassification.from_pretrained("roberta-large-mnli")

def get_entailments(sent):
    """
    Calculates the probabilities of three labels:
    [CONTRADICTION, NEUTRAL, ENTAILMENT]
    """
    inputs = tokenizer(sent, return_tensors="pt")
    outputs = model(**inputs)
    logits = outputs.logits #model's predictions before normalization with softmax
    predictions = torch.nn.functional.softmax(logits, dim=-1).detach().numpy()[0]
    return predictions

def predict_and_record(input_data_df,counter,offset):
    # put sentences together and get prediction
    sent1 = input_data_df.iat[counter+offset, 4]
    sent2 = input_data_df.iat[counter+offset, 5]
    sent = sent1+"</s><s/>"+sent2
    probs = get_entailments(sent)

    colPos = offset*3

    #fill row of sent A
    input_data_df.iat[counter, 6+colPos] = probs[0]#contradiction
    input_data_df.iat[counter, 7+colPos] = probs[1]#neutral
    input_data_df.iat[counter, 8+colPos] = probs[2]#entailment
    
    #fill row of sent B
    input_data_df.iat[counter+1, 6+colPos] = probs[0]
    input_data_df.iat[counter+1, 7+colPos] = probs[1]
    input_data_df.iat[counter+1, 8+colPos] = probs[2]
    return input_data_df

def record_AdifB(input_data_df,counter):
    #fill row of sent A
    input_data_df.iat[counter, 12] = input_data_df.iat[counter, 6]-input_data_df.iat[counter, 9] #contradictive
    input_data_df.iat[counter, 13] = input_data_df.iat[counter, 7]-input_data_df.iat[counter, 10]#neutral
    input_data_df.iat[counter, 14] = input_data_df.iat[counter, 8]-input_data_df.iat[counter, 11]#entailment
    
    #fill row of sent B
    input_data_df.iat[counter+1, 12] = input_data_df.iat[counter, 6]-input_data_df.iat[counter, 9] #contradictive
    input_data_df.iat[counter+1, 13] = input_data_df.iat[counter, 7]-input_data_df.iat[counter, 10]#neutral
    input_data_df.iat[counter+1, 14] = input_data_df.iat[counter, 8]-input_data_df.iat[counter, 11]#entailment
    return input_data_df

def main():
    # get inputfile name from user
    input_file_name= input("Enter name of input sentences, with respect to current directory: ")
    sentence_entailment_file = open(input_file_name)
    sentence_entailment_file.readline()

    # make dataframe from input data
    input_data_df = pd.read_csv(sentence_entailment_file, delimiter='\t', names=['Sentence_Number', 'Version_(A/B)','Task_Name', 'Data_Set', 'Sentence_1', 'Sentence_2','Contradiction_Score_A', 'Neutral_Score_A', 'Entailment_Score_A','Contradiction_Score_B', 'Neutral_Score_B', 'Entailment_Score_B', 'Dif_Contradiction_Score_A.B', 'Neutral_Score_A.B', 'Entailment_Score_A.B'])

    num_rows=len(input_data_df)
    # count no. of lines
    print("Number of lines present: ", len(input_data_df))

    # loop through lines, get predictions, and add to dataframe
    for counter in range(0,num_rows,2):
        input_data_df = predict_and_record(input_data_df,counter,0)
        input_data_df = predict_and_record(input_data_df,counter,1)
        input_data_df = record_AdifB(input_data_df,counter)

    print(input_data_df)

    # output results to file
    output_file_name = input("Enter name of empty output csv, with respect to current directory: ") #relatedness_output_example.csv
    input_data_df.to_csv(output_file_name)

main()
