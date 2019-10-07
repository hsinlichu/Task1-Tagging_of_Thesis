from torchvision import datasets, transforms
from base import BaseDataLoader
from torch.utils.data import Dataset
import pickle


class MnistDataLoader(BaseDataLoader):
    """
    MNIST data loading demo using BaseDataLoader
    """
    def __init__(self, data_dir, batch_size, shuffle=True, validation_split=0.0, num_workers=1, training=True):
        trsfm = transforms.Compose([
            transforms.ToTensor(),
            transforms.Normalize((0.1307,), (0.3081,))
        ])
        self.data_dir = data_dir
        self.dataset = datasets.MNIST(self.data_dir, train=training, download=True, transform=trsfm)
        super().__init__(self.dataset, batch_size, shuffle, validation_split, num_workers)



class ThesisTaggingDataLoader(BaseDataLoader):
    """
    ThesisTagging data loading demo using BaseDataLoader
    """
    def __init__(self, train_data_path, test_data_path, batch_size, shuffle=True, validation_split=0.0, num_workers=1, training=True):
        if training:
            data_path = train_data_path
        else:
            data_path = test_data_path
            
        print(data_path)
        self.dataset = ThesisTaggingDataset(data_path)
        super().__init__(self.dataset, batch_size, shuffle, validation_split, num_workers)




class ThesisTaggingDataset(Dataset):
    def __init__(self, data_path):
        self.data_path = data_path
        with open(data_path, "rb") as f:
            self.dataset = pickle.load(f)

        print("Dataset total length: {}".format(len(self.dataset)))
        
    def __len__(self):
        return len(self.dataset)

    def __getitem__(self, index):
        data = self.dataset[index] 
        return data["sentence"], data["label"]

