import numpy as np
import csv

class arcene_loader:

    # Read files
    def file_reader(self, file_path):
        '''Input = file path (str)
           Output = numpy array of items in files
        '''

        data = []
        with open(file_path, 'r') as f:
            reader = csv.reader(f, delimiter='\n')
            for row in reader:
                for x in row:
                    x = x.split(' ')
                    example = []
                    for item in x:
                        if item:
                            item = int(item)  # convert to int
                            example.append(item)
                    data.append(example)
            data = np.asarray(data)
        return data

    def createArcene(self):
        valid_data = self.file_reader('datasets/arcene/arcene_valid.data')
        train_data = self.file_reader('datasets/arcene/arcene_train.data')
        test_data = self.file_reader('datasets/arcene/arcene_test.data')
        arcene = np.concatenate([valid_data, test_data, train_data])
        np.random.shuffle(arcene)
        return arcene

