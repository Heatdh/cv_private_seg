import torch
import torch.nn as nn
import wandb
from src.dataloder import CancerDataset
from src.model import Net
import pandas as pd 
import os
from PIL import Image
from sklearn.model_selection import train_test_split
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms, utils

class Train_cfgs():
    def __init__(self, args_dict,cfgname):
        self.args = args_dict
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.runname = cfgname[:-5]
        self.best_val_loss = 0.3
        print("training with cfg", args_dict)
        print()
    
    def config_wb(self):
        print("configuring wandb")
        # generate unique run
        unique_id = wandb.util.generate_id()
        wandb.init(unique_id,name= self.runname, project="cancer_cv",entity="heatdh", config=self.args)
        
    def default_data_prep(self):
        DATA_DIR = 'data'
        Image_dir = os.path.join(DATA_DIR, 'jpeg')
        CSV_dir = os.path.join(DATA_DIR, 'csv')
        # please for visualization purposes, use the notebook that shows a walk through of the data
        # we dont want anything to break the pipline
        train_data_meta = pd.read_csv(os.path.join(CSV_dir, 'calc_case_description_train_set.csv'))
        image_path = train_data_meta['image file path'][3].split('/')[-2]
        folder_path = [os.path.join(Image_dir,i.split('/')[-2]) for i in train_data_meta['image file path']]
        train_data_meta['raw_image_path'] = [os.path.join(i,os.listdir(i)[0]) for i in folder_path]
        mapping = {'BENIGN': 0, 'BENIGN_WITHOUT_CALLBACK': 0, 'MALIGNANT': 1, 'MALIGNANT_WITHOUT_CALLBACK': 1}
        train_df_filtered = train_data_meta.drop(['image file path', 'cropped image file path', 'ROI mask file path'], axis=1)
        train_df_filtered['pathology'] = train_df_filtered['pathology'].map(mapping)
        # create train and test data set with target and raw image path read with pillow
        train_df= train_df_filtered[['pathology','raw_image_path']]
        # copy train_df and concatinated with the train_df with the same image path but with mask image path
        # split the data into train and test
        train_df, val_df = train_test_split(train_df, test_size=0.2, random_state=42)
        # create a transform for the data
        transform = transforms.Compose([
            transforms.Resize((224,224)),
            transforms.ToTensor(),
            transforms.Normalize((0.5), (0.5))])


        # create the train and validation dataset
        self.train_dataset = CancerDataset(train_df, transform=transform)
        self.val_dataset = CancerDataset(val_df, transform=transform)


        
    def data_loader(self):
        # create train and test data loader
        self.train_loader = DataLoader(self.train_dataset,self.args["batch_size"], self.args["num_workers"])
        self.val_loader = DataLoader(self.val_dataset,self.args["batch_size"], self.args["num_workers"])

    def loss_set(self):
        # create loss function
        if self.args["loss"] == "cross_entropy":
            self.loss_fn = nn.CrossEntropyLoss()
        elif self.args["loss"] == "bce":
            self.loss_fn = nn.BCELoss()
        elif self.args["loss"] == "categorical_crossentropy":
            self.loss_fn = nn.CategoricalCrossEntropy()
        else:
            raise ValueError("loss function not supported")
    def optimizer_set(self):
        if self.args["optimizer"] == "adam":
            self.optimizer = torch.optim.Adam(self.model.parameters(), lr=self.args["learning_rate"])
        elif self.args["optimizer"] == "sgd":
            self.optimizer = torch.optim.SGD(self.model.parameters(), lr=self.args["learning_rate"])
        else:
            raise ValueError("optimizer not supported")
    
    def model_set(self):
        # create model
        self.model = Net(2,enc=self.args["encoder"], pretrained=True)
        self.model.to(self.device)
        wandb.watch(self.model, log="all")
    
    # train and val one epoch
    def train_one_epoch(self):
        self.model.train()
        train_loss = 0
        for batch_idx, (data, target) in enumerate(self.train_loader):
            data, target = data.to(self.device), target.to(self.device)
            self.optimizer.zero_grad()
            output = self.model(data)
            loss = self.loss_fn(output, target)
            loss.backward()
            self.optimizer.step()
            train_loss += loss.item()
            wandb.log({"train_loss": loss.item()})
        train_loss /= len(self.train_loader.dataset)
        print('Train set: Average loss: {:.4f}'.format(train_loss))
        return train_loss

    def val_one_epoch(self):
        val_loss = 0
        correct = 0
        self.model.eval()
        with torch.no_grad():
            for data, target in self.val_loader:
                data, target = data.to(self.device), target.to(self.device)
                output = self.model(data)
                val_loss += self.loss_fn(output, target).item()
                pred = output.argmax(dim=1, keepdim=True)
                correct += pred.eq(target.view_as(pred)).sum().item()
        val_loss /= len(self.val_loader.dataset)
        print('Val set: Average loss: {:.4f}, Accuracy: {}/{} ({:.0f}%)'.format(
            val_loss, correct, len(self.val_loader.dataset),
            100. * correct / len(self.val_loader.dataset)))
        wandb.log({"val_loss": val_loss, "val_acc": 100. * correct / len(self.val_loader.dataset)})
        return val_loss
    def load_model(self):
        # load model
        self.model.load_state_dict(torch.load(self.args["model_save_dir"]+self.runname+".pt"))
        self.model.eval()

    def train(self):
        # train and val
        # if model exists, load mode"
        if self.runname + ".pt" in os.listdir(self.args["model_save_dir"]):
            self.load_model()
            
        else:
            for epoch in range(1, self.args["epochs"] + 1):
                train_loss = self.train_one_epoch()
                val_loss = self.val_one_epoch()
                # save model
                if val_loss < self.best_val_loss:
                    self.best_val_loss = val_loss
                    torch.save(self.model.state_dict(), self.args["model_save_dir"]+"/"+self.runname+".pt")
                    print("model saved")
    


    def predict(self, image_path):
        image = Image.open(image_path)
        image = self.transform(image)
        image = image.unsqueeze(0).to(self.device)
        output = self.model(image)
        pred = output.argmax(dim=1, keepdim=True)
        return pred.item()

    def predict_proba(self, image_path):
        image = Image.open(image_path)
        image = self.transform(image)
        image = image.unsqueeze(0).to(self.device)
        output = self.model(image)
        return output
    
    def quantize(self):
        # quantize model
        self.model_quant = torch.quantization.quantize_dynamic(
            self.model, {nn.Linear}, dtype=torch.qint8
        )
        torch.save(self.model_quant.state_dict(), self.args["model_save_dir"]+"/"+self.runname+"_quantized.pt")
        print("model quantized")

    def export_onnx(self):
        # export onnx model
        dummy_input = torch.randn(1, 3, 224, 224, device=self.args["device"])
        torch.onnx.export(self.model_quant, dummy_input, self.args["model_save_dir"]+"/"+self.runname+".onnx", verbose=True)
        print("model exported")




    



    
      

