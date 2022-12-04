from PIL import Image
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms, utils
import torch
# create a class for the dataset for the train and validation set
class CancerDataset(Dataset):
    def __init__(self, df, transform=None):
        self.df = df
        self.transform = transform

    def __len__(self):
        return len(self.df)

    def __getitem__(self, idx):
        img_name = self.df.iloc[idx, 1]
        image = Image.open(img_name)
        label = self.df.iloc[idx, 0]
        if self.transform:
            image = self.transform(image)
        return image, label

    # create a default transofrm for the data
    def apply_default_transform(self):
        transform = transforms.Compose([
            transforms.Resize((224,224)),
            transforms.ToTensor(),
            transforms.Normalize((0.5), (0.5))])
        return transform
    # data loader
    def get_data_loader(self,df, batch_size, num_workers, shuffle=True):
        transform = self.apply_default_transform() if self.transform is None else self.transform
        dataset = CancerDataset(df, transform)
        data_loader = DataLoader(dataset=dataset, batch_size=batch_size, shuffle=shuffle, num_workers=num_workers)
        return data_loader

