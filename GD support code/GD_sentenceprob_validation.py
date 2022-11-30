import csv
import pandas as pd
import numpy as np

from transformers import AutoTokenizer, AutoModelForSequenceClassification
from transformers import RobertaForQuestionAnswering, RobertaForMultipleChoice, RobertaForCausalLM
from transformers import RobertaConfig
import torch

#Uses base RoBERTa as model

input_file_name= input("Enter name of input sentences, with respect to current directory: ") #so_dont_I_sentprob.tsv
sentence_probability_file = open(input_file_name)

# make dataframe from input data
input_data_df = pd.read_csv(sentence_probability_file, delimiter='\t', names=['Sentence_Number', 'Version_(A/B)','Task_Name', 'Data_Set', 'Sentence', 'A_score', 'B_score', 'Difference_A-B'])

num_rows=len(input_data_df)
# count no. of lines
print("Number of lines present: ",
      len(input_data_df))
header = ['Sentence_Number', 'Version_(A/B)','Task_Name', 'Data_Set', 'Sentence', 'A_score', 'B_score', 'Difference_A-B'] #header consists label 'Sentence_Number', 'Version_(A/B)','Task_Name', 'Data_Set', 'Sentence'and add three fields. It will give the task score for the first sentence, the task score for the second sentence, and the difference between the two task scores.

sent_A=''
sent_B=''
prob_A=float(-1)
prob_B=float(-1)
dif_A_B = float(-1)
counter =0
# all file inputs will have even number of rows since sentences are in pairs
# On odd i's, we will have currentA= the sentence and sentAprob = model(sentA), and then on the next i,
 # which will be odd, we will have currentA already set (dont change it)
 # and then set currentB to the current sentence.
 # after that, the comparison is available. I will add two rows to output csv (w/ tabs to seperate) once I have the comparison value
 # Then, the if i is odd block will come to an end and loop back to top, to enter the A if on the next i, which should be even.

def get_sentence_probability(sent):
    """
    Feeds a single prompt into RoBERTa and returns its cross-entropy loss
    """
    tokenizer = AutoTokenizer.from_pretrained("roberta-base")
    config = RobertaConfig.from_pretrained("roberta-base")
    config.is_decoder = True
    model = RobertaForCausalLM.from_pretrained("roberta-base", config=config)
    inputs = tokenizer(sent, return_tensors="pt")
    #print("len inputs:",len(inputs["input_ids"]))
    outputs = model(**inputs,labels=inputs["input_ids"])
    tok_context = tokenizer(sent, return_tensors="pt")
    tok_context['input_ids'] = tok_context['input_ids'][:,:-1]
    tok_context['attention_mask'] = tok_context['attention_mask'][:,:-1]
    prior_loss = model(**tok_context,labels=tok_context["input_ids"]).loss.detach().numpy()
    #print("prior loss: ",prior_loss)
    logits = outputs.logits
    last_word_probs = torch.nn.functional.softmax(logits[0][-1])
    predicted_idx = last_word_probs.argmax().item()
    #print("Argmax:",last_word_probs.argmax())
    #print("Logits 0:",last_word_probs)
    #print("Logits:",last_word_probs[predicted_idx])
    loss = outputs.loss.detach().numpy()
    #print("Loss:",loss)
    return(loss)


while counter<num_rows:
    if counter%2==0: #if even
        sent_A = input_data_df.iat[counter, 4]
        prob_A = get_sentence_probability(sent_A)
        print(sent_A)
        print(prob_A)
        input_data_df.iat[counter, 5]=prob_A

        counter=counter+1#now counter to even
        sent_B = input_data_df.iat[counter, 4]
        prob_B = get_sentence_probability(sent_B)
        print(sent_A)
        print(prob_A)
        print(dif_A_B)
        dif_A_B = prob_A-prob_B
        input_data_df.iat[counter-1, 5]=prob_A #input probA in even
        input_data_df.iat[counter-1, 6]=prob_B #input probB in even
        input_data_df.iat[counter-1, 7]=dif_A_B #input probB in even

        input_data_df.iat[counter, 5]=prob_A #input probA in even
        input_data_df.iat[counter, 6]=prob_B #input probB in even
        input_data_df.iat[counter, 7]=dif_A_B #input probB in even
        counter=counter+1#now counter to odd, go to next round

print(input_data_df)
output_file_name = input("Enter name of empty output csv, with respect to current directory: ") #relatedness_output_example.csv
input_data_df.to_csv(output_file_name)

