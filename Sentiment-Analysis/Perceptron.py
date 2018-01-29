import sys
import getopt
import os
import math
import operator
import numpy as np
import time

class Perceptron:
  class TrainSplit:
    """Represents a set of training/testing data. self.train is a list of Examples, as is self.test. 
    """
    def __init__(self):
      self.train = []
      self.test = []

  class Example:
    """Represents a document with a label. klass is 'pos' or 'neg' by convention.
       words is a list of strings.
    """
    def __init__(self):
      self.klass = ''
      self.words = []


  def __init__(self):
    """Perceptron initialization"""
    self.numFolds = 10
    self.num_ex=0
    self.dic={}
    self.weights=np.array([])
    self.train_y=[]

  #############################################################################
  # TODO TODO TODO TODO TODO 
  # Implement the Perceptron classifier
  

  def classify(self, words):
    """ TODO
      'words' is a list of words to classify. Return 'pos' or 'neg' classification.
    """
    ndic={}
    for w in words:
      if w in ndic:
        ndic[w]+=1
      else:
        ndic[w]=1
    x=[]
    for k in self.dic:
      if k in ndic:
        x.append(ndic[k])
      else:
        x.append(0)
    sign=sum(x*self.weights)
    if sign>0:
      return 'pos'
    else:
      return 'neg'

  

  def addExample(self, klass, words):
    """
     * TODO
     * Train your model on an example document with label klass ('pos' or 'neg') and
     * words, a list of strings.
     * You should store whatever data structures you use for your classifier 
     * in the Perceptron class.
     * Returns nothing
    """
    noise=["It","it","a","this","that","I","is","are","They","his","her","movie"]
    words=[word for word in words if word not in noise]
    self.num_ex+=1
    if klass=='pos':
      self.train_y.append(1)
    else:
      self.train_y.append(-1)
    if not self.dic:
      for w in words:
        if w in self.dic:
          self.dic[w][-1]+=1
        else:
          self.dic[w]=[1]
    else:
      for k in self.dic:
        self.dic[k].append(0)
      for w in words:
        if w in self.dic:
          self.dic[w][-1]+=1
        else:
          a=[0]*self.num_ex
          a[-1]+=1
          self.dic[w]=a
    
  
  def train(self, split, iterations):
      """
      * TODO 
      * iterates through data examples
      * TODO 
      * use weight averages instead of final iteration weights
      """
      for example in split.train:
          words = example.words
          self.addExample(example.klass, words)
#      for k in Perceptron.dic:
#        print k
#        print Perceptron.dic[k][:3]
      startTime = time.time()
      num_feature=len(self.dic)
      self.weights=np.zeros(num_feature)
      x=[]
      for k in self.dic:
        x.append(self.dic[k])
      train_x=np.array([np.array(xi) for xi in x]).transpose()
      b=False
      for k in range(iterations):
        for ix in range(self.num_ex):
          if b:
            if self.num_ex%2==1:
              i=self.num_ex-ix-1
            else:
              i=self.num_ex-ix
            b=False
          else:
            i=ix
            b=True
            
          sign=sum(train_x[i]*self.weights)
#          print sign
          if(sign*self.train_y[i]<=0):
            self.weights=self.weights+2*self.train_y[i]*train_x[i]
#            print Perceptron.weights[100:200]
      print 'Training took %fs!' % (time.time() - startTime)
#      print Perceptron.weights[100:200]
          

  # END TODO (Modify code beyond here with caution)
  #############################################################################
  
  
  def readFile(self, fileName):
    """
     * Code for reading a file.  you probably don't want to modify anything here, 
     * unless you don't like the way we segment files.
    """
    contents = []
    f = open(fileName)
    for line in f:
      contents.append(line)
    f.close()
    result = self.segmentWords('\n'.join(contents)) 
    return result

  
  def segmentWords(self, s):
    """
     * Splits lines on whitespace for file reading
    """
    return s.split()

  
  def trainSplit(self, trainDir):
    """Takes in a trainDir, returns one TrainSplit with train set."""
    split = self.TrainSplit()
    posTrainFileNames = os.listdir('%s/pos/' % trainDir)
    negTrainFileNames = os.listdir('%s/neg/' % trainDir)
    for fileName in posTrainFileNames:
      example = self.Example()
      example.words = self.readFile('%s/pos/%s' % (trainDir, fileName))
      example.klass = 'pos'
      split.train.append(example)
    for fileName in negTrainFileNames:
      example = self.Example()
      example.words = self.readFile('%s/neg/%s' % (trainDir, fileName))
      example.klass = 'neg'
      split.train.append(example)
    return split


  def crossValidationSplits(self, trainDir):
    """Returns a lsit of TrainSplits corresponding to the cross validation splits."""
    splits = [] 
    posTrainFileNames = os.listdir('%s/pos/' % trainDir)
    negTrainFileNames = os.listdir('%s/neg/' % trainDir)
    #for fileName in trainFileNames:
    for fold in range(0, self.numFolds):
      split = self.TrainSplit()
      for fileName in posTrainFileNames:
        example = self.Example()
        example.words = self.readFile('%s/pos/%s' % (trainDir, fileName))
        example.klass = 'pos'
        if fileName[2] == str(fold):
          split.test.append(example)
        else:
          split.train.append(example)
      for fileName in negTrainFileNames:
        example = self.Example()
        example.words = self.readFile('%s/neg/%s' % (trainDir, fileName))
        example.klass = 'neg'
        if fileName[2] == str(fold):
          split.test.append(example)
        else:
          split.train.append(example)
      splits.append(split)
    return splits
  

def test10Fold(args):
  pt = Perceptron()
  
  iterations = int(args[1])
  splits = pt.crossValidationSplits(args[0])
  avgAccuracy = 0.0
  fold = 0
  for split in splits:
    classifier = Perceptron()
    accuracy = 0.0
    classifier.train(split,iterations)
  
    for example in split.test:
      words = example.words
      guess = classifier.classify(words)
      if example.klass == guess:
        accuracy += 1.0

    accuracy = accuracy / len(split.test)
    avgAccuracy += accuracy
    print '[INFO]\tFold %d Accuracy: %f' % (fold, accuracy) 
    fold += 1
  avgAccuracy = avgAccuracy / fold
  print '[INFO]\tAccuracy: %f' % avgAccuracy
    
    
def classifyDir(trainDir, testDir,iter):
  classifier = Perceptron()
  trainSplit = classifier.trainSplit(trainDir)
  iterations = int(iter)
  classifier.train(trainSplit,iterations)
  testSplit = classifier.trainSplit(testDir)
  #testFile = classifier.readFile(testFilePath)
  accuracy = 0.0
  for example in testSplit.train:
    words = example.words
    guess = classifier.classify(words)
    if example.klass == guess:
      accuracy += 1.0
  accuracy = accuracy / len(testSplit.train)
  print '[INFO]\tAccuracy: %f' % accuracy
    
def main():
  (options, args) = getopt.getopt(sys.argv[1:], '')
  
  if len(args) == 3:
    classifyDir(args[0], args[1], args[2])
  elif len(args) == 2:
    test10Fold(args)

if __name__ == "__main__":
    main()
