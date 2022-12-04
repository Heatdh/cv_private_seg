**Project structure and goals **
In our project, we combine different aspects of privacy, advanced computer vision, augmentation technique, deployment of self trained model on server and providing a frontend to interact with all components.

### Walkthrough the files
#### Computer Vision and efficient edge training
run_experiments: [run_experiments.py](run_experiments.py): 
The run_experiments script itterate over the config files, get the model parameters to train on, launch the training and log the metrics it on [wandb](wandb.ai) as well as quantize and sparse the model

To our [src](src) folder. For exact implementation details refer to:
- [dataloader.py](src/dataloder.py) = dataloader class 
- [model.py](src/model.py) = model architecture 
- [train_cfgs.py](src/train_cfgs.py) = Top g, contains all the implementation of the training and quantization loop
For a quick recap and visualization you can refer to our [notebook](Computer_Vision_advanced.ipynb) 
A small look into one sample of a config settings stored as a json, we find below, where we use the keys and values to set our model parameters
```json
{
    "epochs": 20,
    "batch_size": 32,
    "learning_rate": 0.001,
    "num_workers": 4,
    "optimizer": "adam",
    "loss": "cross_entropy",
    "metrics": ["accuracy"],
    "encoder": "resnet50",
    "validation_split": 0.2,
    "shuffle": true,
    "verbose": 1,
    "augmentation": true,
    "callbacks": [
        {
            "class_name": "ModelCheckpoint",
            "config": {
                "filepath": "model.h5",
                "monitor": "val_loss",
                "save_best_only": true,
                "save_weights_only": false,
                "mode": "auto",
                "save_freq": "epoch"
            }
        }
    ],
    "model_save_dir": "export_model"
}
```

