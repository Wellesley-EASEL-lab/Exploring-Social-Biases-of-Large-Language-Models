import csv
import pandas as pd
import numpy as np

from transformers import AutoTokenizer, AutoModelForSequenceClassification
from transformers import RobertaConfig
import torch

"""
A script for running a batch of paired sentences through a RoBERTa sentiment classification model 
for the Grammatical Diversity CS 232 final project
Author: Skylar Kolisko
Date: 4/4/2022
"""

model_name = "siebert/sentiment-roberta-large-english"
tokenizer = AutoTokenizer.from_pretrained(model_name)
model = AutoModelForSequenceClassification.from_pretrained(model_name)

def get_sentiments(sent):
    """
    Takes a sentence and feeds it through a RoBERTa-based sentiment analysis model
    Outputs the probabilities of two labels: [NEGATIVE, POSITIVE]
    """
    inputs = tokenizer(sent, return_tensors="pt")
    outputs = model(**inputs)
    logits = outputs.logits
    predictions = torch.nn.functional.softmax(logits, dim=-1).detach().numpy()[0]
    # print(sent,"| p(NEG)=",predictions[0],"| p(POS)=",predictions[1])
    return predictions

def predict_and_record(input_data_df,counter,offset):
    sent = input_data_df.iat[counter+offset, 4]
    probs = get_sentiments(sent)
    colcount = offset*2
    input_data_df.iat[counter, 6+colcount] = probs[0]#negative
    input_data_df.iat[counter, 7+colcount] = probs[1]#positive
    input_data_df.iat[counter+1, 6+colcount] = probs[0]#negative
    input_data_df.iat[counter+1, 7+colcount] = probs[1]#positive

    return input_data_df

def record_AdifB(input_data_df,counter):
    input_data_df.iat[counter, 10] = input_data_df.iat[counter, 6]-input_data_df.iat[counter, 8] #negative
    input_data_df.iat[counter, 11] = input_data_df.iat[counter, 7]-input_data_df.iat[counter, 9]#positive
    input_data_df.iat[counter+1, 10] = input_data_df.iat[counter, 6]-input_data_df.iat[counter, 8] #negative
    input_data_df.iat[counter+1, 11] = input_data_df.iat[counter, 7]-input_data_df.iat[counter, 9] #negative
    return input_data_df

def main():
    # Get input file from user
    input_file_name= input("Enter name of input sentences, with respect to current directory: ") #entail_sents.tsv
    sentence_probability_file = open(input_file_name)

    # make dataframe from input data
    input_data_df = pd.read_csv(sentence_probability_file, delimiter='\t', names=['Sentence_Number', 'Version_(A/B)','Task_Name', 'Data_Set', 'Sentence_1', 'LABEL','Negative_Score_A',  'Positive_Score_A','Negative_Score_B',  'Positive_Score_B', 'Dif_Negative_Score_A.B',  'Positive_Score_A.B'])

    # count no. of lines
    num_rows=len(input_data_df) 
    print("Number of lines present: ", len(input_data_df))

    # process sentences and update dataframe
    for counter in range(0,num_rows,2):
        input_data_df = predict_and_record(input_data_df,counter,0)
        input_data_df = predict_and_record(input_data_df,counter,1)
        input_data_df = record_AdifB(input_data_df,counter)

    print(input_data_df)

    #output scores to output file
    output_file_name = input("Enter name of empty output csv, with respect to current directory: ") #relatedness_output_example.csv
    input_data_df.to_csv(output_file_name)

main()

