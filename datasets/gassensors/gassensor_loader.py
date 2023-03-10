import numpy as np
import csv

class gassensor_loader:

    # Read files
    def file_reader(self, file_path):
        '''Input = file path (str)
           Output = numpy array of items in files
        '''

        data = []
        index = 0
        with open(file_path, 'r') as f:
            reader = csv.reader(f, delimiter='\n')
            for row in reader:
                for x in row:
                    index = 0
                    x = x.split(' ')
                    example = []
                    for item in x:
                        if index != 0:  # verificar o index do mes do ds, para desprezar.
                            if item:
                                new_item = item.split(':')
                                item = float(new_item[1])
                                example.append(item)
                        index = index + 1
                    data.append(example)
            data = np.asarray(data)
        return data

    def createGasSensors(self):
        dt1 = self.file_reader('datasets/gassensors/batch1.dat')
        dt2 = self.file_reader('datasets/gassensors/batch2.dat')
        dt3 = self.file_reader('datasets/gassensors/batch3.dat')
        dt4 = self.file_reader('datasets/gassensors/batch4.dat')
        dt5 = self.file_reader('datasets/gassensors/batch5.dat')
        dt6 = self.file_reader('datasets/gassensors/batch6.dat')
        dt7 = self.file_reader('datasets/gassensors/batch7.dat')
        dt8 = self.file_reader('datasets/gassensors/batch8.dat')
        dt9 = self.file_reader('datasets/gassensors/batch9.dat')
        dt10 = self.file_reader('datasets/gassensors/batch10.dat')
        gassensor = np.concatenate([dt1, dt2, dt3, dt4, dt5, dt6, dt7, dt8, dt9, dt10])
        np.random.shuffle(gassensor)
        return gassensor

