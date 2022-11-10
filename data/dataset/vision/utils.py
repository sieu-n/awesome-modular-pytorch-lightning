from torchvision.datasets import ImageFolder
from pathlib import Path
import os

class ImagesInsideFolder(ImageFolder):
    def __init__(self, root, *args, **kwargs):
        """
        Conduct dataset from images inside folder. Implemented using trick to create ImageFolder
        one directory above with the folder name as the only class.
        """
        self.folder_name = Path(root).stem
        super().__init__(Path(root).parents[0], *args, **kwargs)
    
    def find_classes(self, root):
        return ([self.folder_name], {self.folder_name: -1})
