# phrase sentence class
class MyPhraseSentences(object):
    def __init__(self, fileList):
        self.fileList = fileList

    def __iter__(self):
        for fname in self.fileList:
            for line in open(fname):
                yield line.split()


# sentence class
class MySentences(object):
    def __init__(self, fileList):
        self.fileList = fileList

    def __iter__(self):
        for fname in self.fileList:
            for line in open(fname):
                yield line[2:-3].split()

