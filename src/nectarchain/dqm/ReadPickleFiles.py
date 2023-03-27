import pickle
import sys


filename = sys.argv[1]

infile = open(filename, "rb")
new_dict = pickle.load(infile)
infile.close()
print(new_dict)
