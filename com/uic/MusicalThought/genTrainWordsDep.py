__author__ = 'vignesh'
import ConfigParser as cp
from nltk.tokenize import sent_tokenize
import pickle


from jsonrpc import ServerProxy, JsonRpc20, TransportTcpIp
from simplejson import loads



# Initialize settings for connecting to the stanford corenlp



server = ServerProxy(JsonRpc20(),TransportTcpIp(addr=("127.0.0.1", 8080)))

"""
Gets all the dependency parsing features

Input

s : sentence to be parsed using dependency parser
output : Array of dependencies for the sentence
"""

def getDepFeatures(s):

    result = loads(server.parse(s))
    dependencies = []
    sentences = result['sentences']
    for i in range(len(sentences)):
        dep = result['sentences'][i]['dependencies'] # Get the dependencies from the json
        dependencies += dep
    # for d in dependencies:
    #     print d

    return dependencies # Contains dependencies for a given sentence
    #pprint(result)

    #print "Result", result


if __name__ == '__main__':

    # Parse the configuration
    config = cp.RawConfigParser()
    config.read('config.cfg')
    trainfile = config.get('init', 'trainfile')
    full_text = open(trainfile).read() # Gets the complete text

    # Giving the full text doesn't work. So splitting as sentences and passing to the service
    sentences = sent_tokenize(full_text)

    # Get the dependencies
    alldependencies = []
    sentcount = 0

    for s in sentences:
        d = getDepFeatures(s)
        alldependencies.append(d)
        print 'Parsing sentence : ' + str(sentcount)
        sentcount+=1

    print alldependencies[:10]

    # Store the dependencies to a file
    pickle.dump(alldependencies,open('dep.pickle','w'))

