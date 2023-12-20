import os
from collections import defaultdict


# def generate_data(num_samples, array_size):
#     given = np.random.choice(['A', 'B', 'C'], size=(num_samples, array_size))
#     expected = (given == 'B').astype(int) + (2 * (given == 'A').astype(int))
#     return given, expected

def increment(i=0):
    while True: yield i; i += 1

class generateData():

    def __init__(self, folderPath, class_count):
        self.folderPath = folderPath
        self.class_count = class_count

        self.geneToFilePath = {i:os.join(self.filePath, j) for i, j in zip(increment(), os.listdir(self.folderPath))}
        self.filePathToGene = {j:i for i, j in self.geneToFilePath.items()}

        self.weightCurrent = 0.2
        self.weightOld = 0.8

        # change the vars below to dict of lists instead of a dict and list of dicts, to save some space
        self.geneEvaluations = defaultdict(lambda: 0) # could make this a list of size L_H, to store evaluations
        self.chromosomeEvaluations = [] #change this to only store last L_H evaluations - for the forget operator funcitonality

    def getFilePaths(self, chromosome):
        return [self.geneToFilePath[i] for i, j in enumerate(chromosome) if j != 0]
    
    def getStructure(self):
        return len(self.geneToFilePath), self.class_count
    
    def getImageAndLabel(self, filePath):
        with open(filePath, 'rb') as f:
            data = f.read()

        return (data, filePath.split('\\')[-1])
    
    def getImage(self, filePath):
        with open(filePath, 'rb') as f:
            data = f.read()

        return data

    def getChromosome(self, filePaths):
        n, _ = self.getStructure()
        chromosome = [0]*n
        for filePath in filePaths:
            chromosome[self.filePathToGene[filePath]] = 1

        return chromosome

    def geneEvaluation(self, gene, missclassifications, number):
        
        self.geneEvaluations[gene] = sum(
            self.weightCurrent * (missclassifications/number),
            self.weightOld * self.geneEvaluations[gene]
        )

        return self.geneEvaluations[gene]
    
    def chromosomeEvaluation(self, chromosome, missclassifications, number):
        
        sum_ = 0
        
        for gene, (_, j, k) in enumerate(zip(chromosome, missclassifications, number)):
            sum_ += self.geneEvaluation(gene, j, k)

        self.chromosomeEvaluation.append(dict(self.geneEvaluations))
        return sum_/len([i for i in chromosome if i])

    
        