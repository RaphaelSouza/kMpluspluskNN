import numpy as np
import csv

class watersensors_loader:

    # Read files
    def file_reader(self, file_path):
        '''Input = file path (str)
           Output = numpy array of items in files
        '''

        data = []
        with open(file_path, 'r') as f:
            reader = csv.reader(f, delimiter='\n')
            counter = 0
            index = 0
            for row in reader:
                for x in row:
                    index = 0
                    x = x.split(',')
                    example = []
                    for item in x:
                        counter += 1
                        if item:
                            if index != 0:
                                if item != "?":
                                    item = float(item)
                                    example.append(item)
                                else:
                                    item = 0.0
                                    example.append(item)
                        index = index + 1
                    data.append(example)
                    # print(counter)
                    counter = 0
            data = np.asarray(data)
        return data

    def createWaterSensors(self):
        ds = self.file_reader('datasets/DUWWTP/water-treatment.data')
        watersensor = np.concatenate([ds])
        np.random.shuffle(watersensor)
        return watersensor

