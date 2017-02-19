from .helpers import saveToFile, loadFromFile

from shutil import rmtree
from os.path import join, exists

# ____________________________________________________________________LocalFiles
class LocalFiles(object):

    def __init__(self, path):
        self.path = path
        self.local = {}

    def load(self, filename):
        if exists(join(self.path, filename)):
            self.local[filename] = loadFromFile(join(self.path, filename))
        else:
            self.local[filename] = None
            self.save(filename)
        return self.local[filename]

    def get(self, filename):
        if filename not in self.local:
            self.load(filename)
        return self.local[filename]

    def drop(self, filename):
        if filename in self.local:
            del self.local[filename]

    def save(self, filename):
        if filename in self.local:
            saveToFile(self.local[filename], join(self.path, filename))

    def done(self, filename):
        self.save(filename)
        self.drop(filename)

    def delete(self):
        rmtree(self.path)
