import numpy as np
import csv

class slice_loader:

    # Read files
    def file_reader(self, file_path):
        '''Input = file path (str)
           Output = numpy array of items in files
        '''

        data = []
        with open(file_path, 'r') as f:
            reader = csv.reader(f, delimiter='\n')
            counter = 0
            for row in reader:
                for x in row:
                    x = x.split(';')
                    example = []
                    for item in x:
                        counter += 1
                        if item:
                            item = float(item.replace(".", ""))  # convert to int
                            example.append(item)
                    data.append(example)
                    # print(counter)
                    counter = 0
            data = np.asarray(data)
        return data

    def createSlice(self):
        slice = self.file_reader('datasets/slice/slice_localization_data_tratado.csv')
        slice_ds = np.concatenate([slice])
        np.random.shuffle(slice_ds)
        return slice_ds

