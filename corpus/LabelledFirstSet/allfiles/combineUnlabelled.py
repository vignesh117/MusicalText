import os
import pandas
# Read all flies in the unlabelled corpus
path = 'Unlabeled_in_txt/'
files = os.listdir('Unlabeled_in_txt/')
allcontents = []
for f in files:
    data = open(path + f).read()
    allcontents.append(data)

#Finally join the files
finalstring = ''.join(allcontents)

# Write this to a file
outfile = 'unlabelledcorpusfull.txt'
fp = open(outfile,'w')
fp.write(finalstring)
fp.close()