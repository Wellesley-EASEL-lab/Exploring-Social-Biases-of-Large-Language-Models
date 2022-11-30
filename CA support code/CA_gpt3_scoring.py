import sys
import math

def score_condition(conditions):
	print(conditions)
	reference = conditions['F']
	completions = reference[4:16]
	keys = [completions[i] for i in range(0,len(completions),2)]
	reference_probs = [float(completions[i]) for i in range(1,len(completions),2)]
	reference_dict = dict(zip(keys,reference_probs))
	diff_dict = {}
	for key in conditions.keys():
		if key != 'F':
			condition = conditions[key]
			print(condition)
			suggested = [condition[4:16][i] for i in range(0,len(completions),2)]
			probs = [float(condition[4:16][i]) for i in range(1,len(completions),2)]
			cdict = dict(zip(suggested,probs))
			rmlist = []
			for s in cdict.keys():
				if s not in keys:
					rmlist.append(s)
					cdict['OTHER'] += cdict[s]
			for s in rmlist:
				cdict.pop(s)
			diffs = 0
			for s in cdict.keys():
				diffs += abs(reference_dict[s] - cdict[s])
			diff_dict[key] = diffs
	return diff_dict

def score_all_by_condition(diff_list):
	diff_dict = {'A':0,'B':0,'C':0,'D':0,'E':0}
	name_dict = {'A':'Japan','B':'UK','C':'US','D':'Mexico','E':'India'}	
	for d in diff_list:
		for k in d.keys():
			diff_dict[k] += d[k]
	for k in diff_dict.keys():
		print(f"{name_dict[k]}: {diff_dict[k]/len(diff_list)}")	

def main():
	infile = sys.argv[1]
	items = [s.strip().split('\t') for s in open(infile,'r').readlines()]
	conditions = {}
	for item in items:
		if item[0] in conditions:
			conditions[item[0]][item[1]] = item
		else:
			conditions[item[0]] = {item[1]:item}
	score_dicts = [score_condition(conditions[c]) for c in conditions.keys()]
	score_all_by_condition(score_dicts)



main()