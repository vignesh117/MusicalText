import os
import sys

# method to convert doc to txt
try:
    from xml.etree.cElementTree import XML
except ImportError:
    from xml.etree.ElementTree import XML
import zipfile

from pyth.plugins.rtf15.reader import Rtf15Reader
from pyth.plugins.plaintext.writer import PlaintextWriter

"""
Module that extract text from MS XML Word document (.docx).
(Inspired by python-docx <https://github.com/mikemaccana/python-docx>)
"""

WORD_NAMESPACE = '{http://schemas.openxmlformats.org/wordprocessingml/2006/main}'
PARA = WORD_NAMESPACE + 'p'
TEXT = WORD_NAMESPACE + 't'

def get_rtf_text(path):
	"""
	Take the path of an rtf file as an argument and return the text
	"""
	
		
	doc = Rtf15Reader.read(open(path))

	return PlaintextWriter.write(doc).getvalue()

def get_docx_text(path):
    """
    Take the path of a docx file as argument, return the text in unicode.
    """
    document = zipfile.ZipFile(path)
    xml_content = document.read('word/document.xml')
    document.close()
    tree = XML(xml_content)

    paragraphs = []
    for paragraph in tree.getiterator(PARA):
        texts = [node.text
                 for node in paragraph.getiterator(TEXT)
                 if node.text]
        if texts:
            paragraphs.append(''.join(texts))

    return '\n\n'.join(paragraphs)
    
corpusdir = 'fromSahisnu/'
files = os.listdir(corpusdir)
resultdir = 'processed/'

# Process files in the corpusdir
for f in os.listdir(corpusdir):
	print f
	
	if 'docx' in f:
		# Covert docx to to txt
		txt = get_docx_text(corpusdir + f)
		
		# write the text output file
		outfilename = f.replace(' ','')
		outfilename = outfilename.replace('docx','txt')
		
		outfile = open(resultdir + outfilename, 'w')
		outfile.write(txt)
		
		print 'Processed file : ' + outfilename
		outfile.close()
		
	elif 'rtf' in f:
		
		# convert rtf to txt
		
		txt = get_rtf_text(corpusdir + f)
		
		# write the text output file
		outfilename = f.replace(' ','')
		outfilename = outfilename.replace('rtf','txt')
		
		outfile = open(resultdir + outfilename, 'w')
		outfile.write(txt)
		
		print 'Processed file : ' + outfilename
		outfile.close()
		
	else:
		
		# Just convert the plain text file by changin the file name
		# write the text output file
		outfilename = f.replace(' ','')
		
		outfile = open(resultdir + outfilename, 'w')
		outfile.write(open(corpusdir + f).read())
		
		print 'Processed file : ' + outfilename
		outfile.close()
		
		
# Make all the 10 corpuses

if os.listdir(resultdir) == []:
	raise 'No files in the result directory'
	sys.exit(0)
	
allcontents = []
for f in os.listdir(resultdir):
	
	content = open(resultdir + f).read()
	content = content.replace('\r','')
	allcontents.append(content)
	
completetxt = '\n'.join(allcontents)
outfile = open('allcorpuses.txt','w')
outfile.write(completetxt)
outfile.close()
	

		
		
		
