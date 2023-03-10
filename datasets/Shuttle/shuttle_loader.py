import numpy as np
import csv

class shuttle_loader:

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
                            item = float(item)  # convert to int
                            example.append(item)
                    data.append(example)
            data = np.asarray(data)
        return data

    def createShuttle(self):
        full_data = self.file_reader('datasets/Shuttle/shuttle.trn')
        shuttle = np.concatenate([full_data])
        np.random.shuffle(shuttle)
        return shuttle

