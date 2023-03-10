import numpy as np
import csv

class multiplefeatures_loader:

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
                    x = x.split('  ')
                    example = []
                    for item in x:
                        counter += 1
                        if item:
                            item = float(item)  # convert to int
                            example.append(item)
                    data.append(example)
                    print(counter)
                    counter = 0
            data = np.asarray(data)
        return data

    def createmultiplefeature(self):
        #fac = self.file_reader('datasets/Multiplefeatures/mfeat-fac.data')
        fou = self.file_reader('datasets/Multiplefeatures/mfeat-fou.data')
        #kar = self.file_reader('datasets/Multiplefeatures/mfeat-kar.data')
        #mor = self.file_reader('datasets/Multiplefeatures/mfeat-mor.data')
        #pix = self.file_reader('datasets/Multiplefeatures/mfeat-pix.data')
        #zer = self.file_reader('datasets/Multiplefeatures/mfeat-zer.data')
        multiplefeature = np.concatenate([fou])
        np.random.shuffle(multiplefeature)
        return multiplefeature

