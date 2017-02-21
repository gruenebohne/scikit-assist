# -*- coding: utf-8 -*-

from .helpers import saveToFile, loadFromFile

from shutil import rmtree
from os.path import join, exists

# ____________________________________________________________________LocalFiles
class LocalFiles(object):
    """Base class for managing files in `path`.

    No files are loaded by default. Files must be loaded through the `load()` 
    function, after which they are availible through `self.local[filename]`. The
    functions `drop()`, `save()` and `done()` are used to drop a file or save it
    to disk, or both.

    Attributes:
        path (:obj:str): Path to the managed folder.

        local (:obj:dict): Dictionary holding the loaded files with key: `filename`.

    .. todo::
        * exception handling and throwing
        * implement `list_files()`
        * implement `with LocalFile.load(filename)`
        * implement `with LocalFile.load([filenames])`

    """

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
