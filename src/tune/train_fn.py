import os
import sys
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "..")))
import tempfile
from pathlib import Path
import torch
from ray import train
from ray.train import Checkpoint, get_checkpoint
import ray.cloudpickle as pickle
from src.train import loss_function
import logging
from src.dataset import Load_Dataset
from src.utils import calculate_accuracy
from src.model import SimpleCNN

def train_cifar(config,data_path,device ='cuda'):

    logger = logging.getLogger(__name__)
    model = SimpleCNN()
    model = model.to(device)
    # optimizer_name = config.get("optimizer", "adam")
    # if optimizer_name == "adam":
    #     optimizer = torch.optim.Adam(
    #         model.parameters(),
    #         lr=lr,
    #         weight_decay=weight_decay
    #     )
    # elif optimizer_name == "sgd":
    #     optimizer = torch.optim.SGD(
    #         model.parameters(),
    #         lr=lr,
    #         weight_decay=weight_decay
    #     )
    # else:
    #     raise ValueError(f"Unsupported optimizer: {optimizer_name}")
    optimizer = torch.optim.Adam(model.parameters(), lr=config['lr'], weight_decay=config.get("weight_decay", 0.0))
    batch_size = config.get("batch_size", 64)
    criterion = torch.nn.CrossEntropyLoss()
    _,train_loader,val_loader,__ = Load_Dataset(train_data_path= data_path,batch_size= batch_size)


    checkpoint = get_checkpoint()
    if checkpoint:
        with checkpoint.as_directory() as checkpoint_dir:
            data_path = Path(checkpoint_dir) / "data.pkl"
            with open(data_path, "rb") as fp:
                checkpoint_state = pickle.load(fp)
            start_epoch = checkpoint_state["epoch"]
            model.load_state_dict(checkpoint_state["net_state_dict"])
            optimizer.load_state_dict(checkpoint_state["optimizer_state_dict"])
    else:
        start_epoch = 0


    for epoch in range(start_epoch,10):
        running_loss = 0.0
        epoch_loss = 0.0 
        epoch_steps = 0
        train_count= 0
        for idx,data in enumerate(train_loader):
            train_inputs,train_labels = data
            train_inputs,train_labels = train_inputs.to(device),train_labels.to(device)
            loss,prediction = loss_function(model=model,criterion=criterion,input =train_inputs,label = train_labels,optim = optimizer)
            running_loss += loss
            epoch_loss+= loss
            epoch_steps +=1
            train_count += train_labels.size(0)
            if idx % 2000 ==1999:
                logger.info(
                    "[%d, %5d] loss: %.3f"  
                    % (epoch + 1, idx + 1, running_loss / epoch_steps)
                )
                running_loss = 0.0
        epoch_loss /= train_count
        logger.info(f"train : Epoch [{epoch+1}/10], train Loss: {epoch_loss:.4f}")

        # Validation loss
        
        total_loss = 0.0
        # val_steps = 0
        val_count = 0
        correct = 0
        with torch.no_grad():
            for idx,data in enumerate(val_loader):
                val_inputs,val_labels = data
                val_inputs,val_labels = val_inputs.to(device),val_labels.to(device)
                loss,prediction = loss_function(model=model,criterion=criterion,input =val_inputs,label = val_labels)
                total_loss+= loss
                correct  += calculate_accuracy(prediction,val_labels)
                val_count += val_labels.size(0)
                # val_steps +=1
            acc = correct / val_count
            val_loss =  total_loss / val_count
            logger.info(f"val : Epoch [{epoch+1}/{10}], val Loss: {val_loss:.4f}, val Acc: {acc:.4f}")
            
        checkpoint_data = {
            "epoch": epoch,
            "net_state_dict": model.state_dict(),
            "optimizer_state_dict": optimizer.state_dict(),
        }

        with tempfile.TemporaryDirectory() as checkpoint_dir:
            data_path = Path(checkpoint_dir) / "data.pkl"
            with open(data_path, "wb") as fp:
                pickle.dump(checkpoint_data, fp)

            checkpoint = Checkpoint.from_directory(checkpoint_dir)
            train.report(
                {"loss": val_loss , "accuracy": acc},
                checkpoint=checkpoint,
            )
