import os
import sys
import math
import openai
import csv

openai.api_key = "INSERT_API_KEY_HERE"

def query(text,n=1):
  response = openai.Completion.create(
  engine="text-davinci-002",
  prompt=text,
  temperature=0.5,
  max_tokens=10,
  top_p=1,
  n=n,
  frequency_penalty=0,
  presence_penalty=0,
  logprobs=5,
  stop="."
)
  return response["choices"]

def addOther(log_prob_dict):
  """Make a log prob dict a probability distribution by adding OTHER category"""
  total = sum([math.exp(v) for v in log_prob_dict.values()])
  other_prob = 1 - total
  log_prob_dict["OTHER"] = math.log(other_prob)
  assert sum([math.exp(v) for v in log_prob_dict.values()]) == 1 #Some probability has been lost if this fails
  
  return log_prob_dict

def stripKeys(prob_dict):
  keys = list(prob_dict.keys())
  for k in keys:
    key = k.strip()
    if "\n" in key:
      key = key.replace("\\n","\n")
    prob_dict[key] = prob_dict[k]
    prob_dict.pop(k)
  return prob_dict

def run_one_item(textprompt,n):
  responses = query(textprompt,n=n)
  texts = [r["text"].strip().split('\n')[0].strip() for r in responses]
  print(texts)
  log_prob_dicts = [r["logprobs"]["top_logprobs"][0] for r in responses]
  log_prob_dicts = [addOther(d) for d in log_prob_dicts]
  log_prob_dicts = [stripKeys(d) for d in log_prob_dicts]
  print([log_prob_dict.keys() for log_prob_dict in log_prob_dicts])

  all_log_probs = [i for d in log_prob_dicts for i in d.items()]
  new_probs = {}
  for k,v in all_log_probs: #Add probabilities across samples
    if k in new_probs:
      new_probs[k] += math.exp(v)
    else:
      new_probs[k] = math.exp(v)

  norm = sum(new_probs.values()) #Re-normalize 
  for k,v in new_probs.items(): 
    new_probs[k] = new_probs[k]/norm

  return new_probs,texts

def export(fn,item_list,prob_dicts,texts):
  with open(fn+"_results.tsv",'w') as of:
    writer = csv.writer(of, delimiter='\t')
    for i,item in enumerate(item_list):
      problist = [item for k in sorted(prob_dicts[i].keys()) for item in [k,prob_dicts[i][k]]]
      bits = item+problist+texts[i]
      line = [str(b) for b in bits]
      writer.writerow(line)

def main():
  infile = sys.argv[1]
  outfile = infile.split('.')[0]
  items = [s.strip().split('\t') for s in open(infile,'r').readlines()]
  results = [run_one_item(item[2],5) for item in items]
  probs = [r[0] for r in results]
  texts = [r[1] for r in results]
  export(outfile,items,probs,texts)
  

main()