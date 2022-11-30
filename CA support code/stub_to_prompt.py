import sys
"""
Script for taking a list of prompts and creating different city variants.
"""

def main():
	cities = ["Tokyo","London","New York","Mexico City","Mumbai","QQQ"]
	labels = ["Japan","UK","US","Mexico","India","Neutral"]
	letters = "ABCDEF"
	stubs = [l.strip() for l in open(sys.argv[1],'r').readlines()]
	for s,stub in enumerate(stubs):
		for c,city in enumerate(cities):
			fields = [s,letters[c],stub.replace('QQQ',city),labels[c]]
			print('\t'.join([str(f) for f in fields]))

main()