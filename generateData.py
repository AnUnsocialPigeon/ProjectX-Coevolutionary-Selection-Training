import os
from functools import reduce
from collections import defaultdict
from itertools import count


# def generate_data(num_samples, array_size):
#     given = np.random.choice(['A', 'B', 'C'], size=(num_samples, array_size))
#     expected = (given == 'B').astype(int) + (2 * (given == 'A').astype(int))
#     return given, expected


class generateData():

    def __init__(self, folderPath, classCount):
        self.folderPath = folderPath
        self.classCount = classCount

        self.geneToFilePath = {i:os.join(self.filePath, j) for i, j in zip(count(), os.listdir(self.folderPath))}
        self.filePathToGene = {j:i for i, j in self.geneToFilePath.items()}

        self.weightCurrent = 0.2
        self.weightOld = 0.8

        # change the vars below to dict of lists instead of a dict and list of dicts, to save some space
        self.geneEvaluations = defaultdict(lambda: 0) # could make this a list of size L_H, to store evaluations
        self.chromosomeEvaluations = [] # forget operator functionality

    def getFilePaths(self, chromosome):
        return [self.geneToFilePath[i] for i, j in enumerate(chromosome) if j != 0]
    
    def getStructure(self):
        return len(self.geneToFilePath), self.classCount
    
    def getImageAndLabel(self, filePath): #lots of assumptions made on how ill get data
        with open(filePath, 'rb') as f:
            data = f.read()

        return (data, filePath.split('\\')[-1])
    
    def getImage(self, filePath): #lots of assumptions made on how ill get data
        with open(filePath, 'rb') as f:
            data = f.read()

        return data

    def getChromosome(self, filePaths):
        n, _ = self.getStructure()
        chromosome = [0]*n
        for filePath in filePaths:
            chromosome[self.filePathToGene[filePath]] = 1

        return chromosome

    def addGeneEvalHistory(self, geneDict):
        self.chromosomeEvaluations.append(geneDict)
        if len(self.chromosomeEvaluations) > self.L_H:
            self.chromosomeEvaluations.pop(0)

    def geneEvaluation(self, gene, missclassifications, number):
        
        self.geneEvaluations[gene] = sum(
            self.weightCurrent * (missclassifications/number),
            self.weightOld * self.geneEvaluations[gene]
        )

    def chromosomeEvaluation(self, chromosome):        
        sum_ = reduce(lambda a, b: self.geneEvaluation[a] + b, [i for i, j in chromosome if j], 0)
        return sum_/sum(chromosome)

    
        