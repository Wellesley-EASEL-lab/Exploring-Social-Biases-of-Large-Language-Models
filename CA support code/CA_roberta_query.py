import os
import sys
import math
from transformers import RobertaTokenizer, RobertaForMaskedLM
import torch

"""
Takes a TSV file of prompts and assesses the probability of each sentence
with different fill-in-the-blank options.

Anything after column 4 in the TSV is treated as a completion option.

This program also retrieves the most likely fill-in-the-blank word
according to RoBERTa, along with the probability of the sentence when
completed that way.

Writes out a TSV file containing the fill-in-blank options assessed and 
the probability of the resulting sentence according to RoBERTa.

With small tweaks, you could retrieve the model loss instead.

Author: Carolyn Anderson
Date: 4/29/22
"""

tokenizer = RobertaTokenizer.from_pretrained("roberta-base")
model = RobertaForMaskedLM.from_pretrained("roberta-base")

def get_most_likely_cloze(textprompt):
  """get the most likely fill-in-the-blank option according to RoBERTa"""
  print(textprompt)
  before,after = textprompt.split('BLANK')
  text = before+'<mask>'+after
  inputs = tokenizer(text, return_tensors="pt")
  with torch.no_grad():
    logits = model(**inputs).logits
  mask_token_index = (inputs.input_ids == tokenizer.mask_token_id)[0].nonzero(as_tuple=True)[0]
  predicted_token_id = logits[0, mask_token_index].argmax(axis=-1)
  most_likely_word = tokenizer.decode(predicted_token_id)
  most_likely_prob = logits[0, mask_token_index].amax(axis=-1).item()
  return (most_likely_word.strip(),most_likely_prob)

def assess_cloze_probability(textprompt,choices):
  """assess fill-in-blank probability of all choices"""
  probs = []
  before,after = textprompt.split('BLANK')
  for c in choices:
    c_idx = tokenizer(' '+c+' ')['input_ids']
    c_len = len(c_idx)-3
    text = before+'<mask> '*(c_len-1)+'<mask>'+after
    inputs = tokenizer(text, return_tensors="pt")

    labels = tokenizer(before+c+after, return_tensors="pt")["input_ids"]
    labels = torch.where(inputs.input_ids == tokenizer.mask_token_id, labels, -100)
    
    outputs = model(**inputs, labels=labels)
    loss = outputs.loss.item()
    probs.append(math.exp(-loss)) # loss is negative log likelihood. so, multiply by -1 and apply inverse of log function
  return probs

def export(fn,item_list,best_words,results):
  """export results"""
  with open(fn+"_results.tsv",'w') as of:
    for i,item in enumerate(item_list):
      item += [best_words[i]]
      result = results[i]
      problist = [item for sublist in zip(item[4:],result) for item in sublist]
      bits = item[:4]+problist
      line = "\t".join([str(b) for b in bits])+'\n'
      of.write(line)

def main():
  infile = sys.argv[1]
  outfile = infile.split('.')[0]
  items = [s.strip().split('\t') for s in open(infile,'r').readlines()]
  print(items[0])
  results = []
  best_words = []
  for item in items:
    best_word,_ = get_most_likely_cloze(item[3])
    best_words.append(best_word)
    result = assess_cloze_probability(item[3],item[4:]+[best_word])
    results.append(result)
  print(best_words)
  export(outfile,items,best_words,results)
  
main()