from numpy import *

class bayes(object):
    """
               p(w|Ci)p(Ci)
    p(Ci|w) = ---------------
                   p(w)
    """
    
    def loadDataSet(self):
        data = [
                ['my','dog','has','flea','problems','help','please'],
                ['maybe','not','take','him','to','dog','park','stupid'],
                ['my','dalmation','so','cute','I','love','him'],
                ['stop','posting','stupid','garbage','worthless'],
                ['mr','licks','ate','my','steak','how','to','stop','him'],
                ['quit','buying','worthless','dog','food','stupid']
            ]
        
        labels = [0, 1, 0, 1, 0, 1]
        return data, labels
    
    def createVocab(self, data):
        vocab = set([])
        for doc in data:
            vocab = vocab | set(doc)
            
        return list(vocab)
    
    def setOfWords2Vec(self, vocab, inputSet):
        vec = [0]*len(vocab)    # Initial a vocab vec and the length is vocab size
        for word in inputSet:
            if word in vocab:
                vec[vocab.index(word)] = 1
            else:
                print('the word %s is not in the vocabulary' % word)
        return vec
    
    def trainNB(self, trainMatrixVec, trainCategory):
        # trainMatrixVec like [[0, 0, 1, 0, 1, 0],
        #                      [0, 0, 0, 0, 1, 0],
        #                        ...
        #                      ]
        numTrainDocs = len(trainMatrixVec)
        wordsEachDoc = len(trainMatrixVec[0])
        pAbusive = sum(trainCategory)/float(numTrainDocs)
        p0Num, p1Num = ones(wordsEachDoc), ones(wordsEachDoc)   # To avoid the probability of some words is 0
        p0Denom, p1Denom = 2.0, 2.0
        for i in range(numTrainDocs):
            if trainCategory[i] == 1:
                p1Num += trainMatrixVec[i]
                p1Denom += sum(trainMatrixVec[i])
            else:
                p0Num += trainMatrixVec[i]
                p0Denom += sum(trainMatrixVec[i])
        
        # Matrix of each word's probability in category
        p0Vec = log(p0Num/p0Denom)      # Use log in order to make the value easy to be distinguished
        p1Vec = log(p1Num/p1Denom)
        
        return p0Vec, p1Vec, pAbusive
    
    def classifyNB(self, inputVec, p0Vec, p1Vec, pAbusive):
        # log(AB) = logA + logB, so actually it is multiplication
        p1 = sum(inputVec * p1Vec) + log(pAbusive)
        p0 = sum(inputVec * p0Vec) + log((1-pAbusive))
        
        print('p1=%s' % p1)
        print('p0=%s' % p0)
        
        if p1 > p0:
            return 1
        else:
            return 0
        
    
    
if __name__ == '__main__':
    bayes = bayes()
    data, labels = bayes.loadDataSet()
    vocab = bayes.createVocab(data)
    mat = []
    for item in data:
        mat.append(bayes.setOfWords2Vec(vocab, item))
        
    p0, p1, pAbusive = bayes.trainNB(mat, labels)   # Training probability
    
    
    test = ['mr', 'dog', 'stupid']
    test_vec = bayes.setOfWords2Vec(vocab, test)
    predict_class = bayes.classifyNB(test_vec, p0, p1, pAbusive)
    
    print(predict_class)
    
    
    
    
    
    
    
    
    
    
    
    