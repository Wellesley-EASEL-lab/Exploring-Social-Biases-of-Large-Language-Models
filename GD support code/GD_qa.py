import csv
import pandas as pd
import numpy as np

from transformers import AutoTokenizer, AutoModelForSequenceClassification
from transformers import RobertaForQuestionAnswering, RobertaForMultipleChoice
from transformers import RobertaConfig
import torch

"""
A script for running a batch of paired sentences through a RoBERTa-based question-answering model 
for the Grammatical Diversity CS 232 final project
Author: Skylar Kolisko
Date: 4/24/2022
"""

tokenizer = AutoTokenizer.from_pretrained("LIAMF-USP/roberta-large-finetuned-race")
model = RobertaForMultipleChoice.from_pretrained("LIAMF-USP/roberta-large-finetuned-race")

def get_multiple_choice_answers(context,question,options):
    questions = []
    for ending_idx, ending in enumerate(options):
        if question.find("_") != -1:
            # fill-in-the-blank questions
            question_option = question.replace("_", ending)
        else:
            question_option = question + " " + ending
        inputs = tokenizer(context,question_option,add_special_tokens=True,padding="max_length",truncation=True,return_overflowing_tokens=False, return_tensors="pt")
        questions.append(question_option)
    encoding = tokenizer([context for o in options], questions, return_tensors="pt", padding=True,truncation=True)
    
    output = model(**{k: v.unsqueeze(0) for k, v in encoding.items()})
    logits = output.logits
    predictions = torch.nn.functional.softmax(logits, dim=-1).detach().numpy()[0]
    predstr = ' '.join([f"| p(Answer {i})={predictions[i]:.4f}" for i in range(len(options))])
    return predictions

def predict_and_record(input_data_df, counter,offset):
    choice_dict = {"A":0, "B":1, "C":2, "D":3 }
    #run model for version A input
    qa = input_data_df.iat[counter+offset, 5]
    contexta = input_data_df.iat[counter+offset, 4]
    opt1 = input_data_df.iat[counter+offset,6]
    opt2 = input_data_df.iat[counter+offset,7]
    opt3 = input_data_df.iat[counter+offset,8]
    opt4 = input_data_df.iat[counter+offset,9]
    optionsa = [opt1, opt2, opt3, opt4]

    # get predictions
    predictions= get_multiple_choice_answers(contexta,qa,optionsa)
    correct  = input_data_df.iat[counter+offset, 10]

    #fill correct for row of qA for version A input
    colPos = offset*4

    #record prediction for qa in row of qa for version A input
    input_data_df.iat[counter, 11+colPos] = predictions[0]#prediction for option a
    input_data_df.iat[counter, 12+colPos] = predictions[1]#prediction for option b
    input_data_df.iat[counter, 13+colPos] = predictions[2]#prediction for option c
    input_data_df.iat[counter, 14+colPos] = predictions[3]#prediction for option d

    #record prediction in row of qb for version A input
    input_data_df.iat[counter+1, 11+colPos] = predictions[0]#predicted for option a
    input_data_df.iat[counter+1, 12+colPos] = predictions[1]#prediction for option b
    input_data_df.iat[counter+1, 13+colPos] = predictions[2]#prediction for option c
    input_data_df.iat[counter+1, 14+colPos] = predictions[3]#prediction for option d

    #record difference between correct and predicted for version A
    highest = max(predictions)
    model_predicts = np.where(predictions == highest)
    if (model_predicts == correct):
        input_data_df.iat[counter, 23+offset] = 0.0
        input_data_df.iat[counter+1, 23+offset] = 0.0
    else:
        input_data_df.iat[counter, 23+offset] = 1.0
        input_data_df.iat[counter+1, 23+offset] = 1.0

    return input_data_df

def record_AdifB(df, counter):
    #record dif for row of question version A
    df.iat[counter, 19] = df.iat[counter, 11] - df.iat[counter, 15] #dif between prediction for opt a with question version a and prediction for opt a with question version b
    df.iat[counter, 20] = df.iat[counter, 12] - df.iat[counter, 16]
    df.iat[counter, 21] = df.iat[counter, 13] - df.iat[counter, 17]
    df.iat[counter, 22] = df.iat[counter, 14] - df.iat[counter, 18]
    #record dif for row of question version B
    df.iat[counter+1, 19] = df.iat[counter+1, 11] - df.iat[counter+1, 15] #dif between prediction for opt a with question version a and prediction for opt a with question version b
    df.iat[counter+1, 20] = df.iat[counter+1, 12] - df.iat[counter+1, 16]
    df.iat[counter+1, 21] = df.iat[counter+1, 13] - df.iat[counter+1, 17]
    df.iat[counter+1, 22] = df.iat[counter+1, 14] - df.iat[counter+1, 18]
    return df

def main():
    # get input file from user
    input_file_name= input("Enter name of input sentences, with respect to current directory: ") #so_dont_I_qa.tsv
    qa_file = open(input_file_name)
    #qa_file.readline()  #uncomment if you have header row
    input_data_df = pd.read_csv(qa_file, delimiter='\t', names=['Sentence_Number', 'Version_(A/B)','Task_Name', 'Data_Set', 'Context', 'Question','Option_A', 'Option_B', 'Option_C', 'Option_D', 'Correct_Answer', 'Context_A_Prediction_A', 'Context_A_Prediction_B','Context_A_Prediction_C', 'Context_A_Prediction_D','Context_B_Prediction_A', 'Context_B_Prediction_B','Context_B_Prediction_C','Context_B_Prediction_D', 'Dif_Scores_A.B_Prediction_A', 'Dif_Scores_A.B_Prediction_B', 'Dif_Scores_A.B_Prediction_C', 'Dif_Scores_A.B_Prediction_D','Was_Prediction_Correct_VersionA','Was_Prediction_Correct_VersionB'])
    
    # count no. of lines
    num_rows=len(input_data_df)
    print("Number of lines present: ", len(input_data_df))

    print(input_data_df)

    # loop through rows and retrieve predictions
    for counter in range(0,num_rows,2):
        input_data_df = predict_and_record(input_data_df, counter, 0)
        input_data_df = predict_and_record(input_data_df, counter, 1)
        input_data_df = record_AdifB(input_data_df, counter)

    print(input_data_df)

    # output data to file
    output_file_name = input("Enter name of empty output csv, with respect to current directory: ") #pt4_out.csv
    input_data_df.to_csv(output_file_name)

main()
