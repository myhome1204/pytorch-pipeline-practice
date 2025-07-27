import os
import sys
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "..")))

import torch
from src.utils import calculate_accuracy , get_gradient_snapshot
import numpy as np
import mlflow
from src.dataset import Sanity_load_Dataset
import copy
import logging



# input을 받아서 loss를 계산해서 optim까지 해서 업데이트까지하는 함수. 그럴라면 loss를 가정해져있어야함
def loss_function(model,criterion,input,label,optim=None):
    # prediction 하나의 샘플에 대한 10개의 클래스별 점수 (logit 또는 확률값)
    prediction = model(input)
    loss = criterion(prediction, label)
    if optim is not None:
        optim.zero_grad()
        loss.backward()
        optim.step()
    return loss.item(), prediction

def fit(
    model,
    train_dataloader,
    val_dataloader,
    criterion,
    optim,
    epochs=100,
    scheduler=None,
    early_stopping = None,
    writer = None
):  
    def log_scalar(writer, tag, value, step):
        if writer:
            writer.add_scalar(tag, value, step)

    def log_mlflow(tag, value, step):
        mlflow.log_metric(tag, value, step=step)
    logger = logging.getLogger(__name__) 
    epoch_gradients = []
    for epoch in range(epochs):
        model.train()  
        total_train_loss = 0.0
        correct_train_acc = 0
        train_count = 0
        batch_gradients  = []

        for idx,data in enumerate(train_dataloader):
            
            train_input, train_label = data
            train_input, train_label = train_input.to(model.device), train_label.to(model.device)
            if idx < 3:  # 처음 3개만
                logger.debug(f"[batch {idx}] x: {train_input.shape}, y: {train_label}")
            
            train_loss,train_prediction = loss_function(model,criterion,train_input,train_label,optim)
            
            grad_snapshot = get_gradient_snapshot(model= model,exclude_bias=False,reduction='mean')
            batch_gradients.append(grad_snapshot)

            total_train_loss += train_loss
            
            acc = calculate_accuracy(train_prediction,train_label)
            correct_train_acc += acc
            train_count += train_label.size(0)

        # epoch 평균 gradient 저장
        epoch_mean_grad = np.mean(batch_gradients, axis=0)
        epoch_gradients.append(epoch_mean_grad)
        for layer_idx, grad in enumerate(epoch_mean_grad):
            log_scalar(writer, f"Gradient/layer_{layer_idx}", grad, epoch)


        total_train_acc = correct_train_acc / train_count
        total_train_loss /= len(train_dataloader) 
        # print(f"Epoch [{epoch+1}/{epochs}], Train Loss: {total_train_loss:.4f}, Train Acc: {total_train_acc:.4f}")
        logger.info(f"train : Epoch [{epoch+1}/{epochs}], Train Loss: {total_train_loss:.4f}, Train Acc: {total_train_acc:.4f}")
        log_scalar(writer, "total_train_acc", total_train_acc, epoch)
        log_scalar(writer, "total_train_loss", total_train_loss, epoch)
        log_mlflow("total_train_acc", total_train_acc, epoch)
        log_mlflow("total_train_loss", total_train_loss, epoch)
    
        model.eval() 
        total_val_loss = 0.0
        correct_val_acc = 0
        val_count = 0
        with torch.no_grad():
            for idx,data in enumerate(val_dataloader):
               
                val_input, val_label = data
                val_input, val_label = val_input.to(model.device), val_label.to(model.device)

                if idx < 3:  # 처음 3개만
                    logger.debug(f"[batch {idx}] x: {val_input.shape}, y: {val_label}")


                val_loss, val_prediction = loss_function(model, criterion, val_input, val_label)
                val_acc = calculate_accuracy(val_prediction,val_label)
                total_val_loss += val_loss
                correct_val_acc +=val_acc
                val_count += val_label.size(0)
                
        total_val_acc = correct_val_acc / val_count
        total_val_loss /= len(val_dataloader)
        logger.info(f"val : Epoch [{epoch+1}/{epochs}], val Loss: {total_val_loss:.4f}, val Acc: {total_val_acc:.4f}")
        log_scalar(writer, "total_val_acc", total_val_acc, epoch)
        log_scalar(writer, "total_val_loss", total_val_loss, epoch)
        log_mlflow("total_val_acc", total_val_acc, epoch)
        log_mlflow("total_val_loss", total_val_loss, epoch)



        # -------------------- scheduler --------------------
        if scheduler:
            scheduler.step(total_val_loss)
            lr = optim.param_groups[0]["lr"]
            logger.info(f"[Epoch {epoch+1}] current_lr : {lr:.6f}")
            log_scalar(writer, "Learning Rate", lr, epoch)

        # -------------------- early stopping --------------------
        if early_stopping:
            early_stopping(score=total_val_loss, model=model, epoch=epoch, optimizer=optim)
            if early_stopping.early_stop:
                logger.warning(f"[EarlyStopping] Stopped early at epoch {epoch+1}")
                break


def sanity_check(model, dataset, criterion, sanity_data_size=64, batch_size=8, device="cpu", epochs=20,log = True):
    sanity_dataloader = Sanity_load_Dataset(dataset,sanity_data_size,batch_size)
    temp_model = copy.deepcopy(model)
    optimizer = torch.optim.Adam(temp_model.parameters(), lr=0.01)
    
    for epoch in range(epochs):
        total_acc = 0.0
        total_loss = 0.0
        count = 0 
        
        for data in sanity_dataloader:
            inputs,labels = data
            inputs,labels = inputs.to(temp_model.device),labels.to(temp_model.device)
            loss , prediction = loss_function(temp_model,criterion,inputs,labels,optimizer)
            acc = calculate_accuracy(prediction,labels)
            total_acc += acc
            total_loss +=loss
            count += labels.size(0)
        avg_acc = total_acc / count
        avg_loss = total_loss / len(sanity_dataloader)
        
        if log:
            mlflow.log_metric("sanity_acc",avg_acc,step = epoch)
            mlflow.log_metric("sanity_loss",avg_loss,step = epoch)
            
        print(f"[Sanity Epoch {epoch+1}/{epochs}] acc={avg_acc:.4f}, loss={avg_loss:.4f}")