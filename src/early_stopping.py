import numpy as np
import logging
import torch

class EarlyStopping:
    def __init__(self,patience = 5 ,delta = 0.0 , mode= 'min',path = "checkpoint.pt"):
        '''
        patience (int) : loss or acc가 개선된 후에 조기종료까지 기다리는 횟수. (Default : 3 )
        delta  (float) : 개선시 인정되는 최소 변화 수치. (Default : 0.0)
        mode     (str) : 개선시 최소/최대 값 기준 선정("min" or "max"). (Defaul : min)
        path     (str) : checkpoint저장 경로 (Default: 'checkpoint.pt')
        '''
        self.early_stop = False
        self.patience = patience
        self.mode = mode
        self.delta = delta
        self.counter = 0 #성능이 개선되지 않은 epoch 횟수를 세는 카운터
        self.best_score = np.Inf if mode == 'min' else 0 # mode에 따라 best값다르게 설정
        self.logger = logging.getLogger(__name__)
        self.path = path

    def __call__(self,score,model,epoch,optimizer):
        if self.mode == 'min': # loss를 최소화.
            if score < (self.best_score-self.delta):
                self.counter = 0
                self.best_score = score
                self.logger.info(f"[EarlyStopping] best score updated to {score:.5f}")
                self.save_checkpoint(model,score,self.best_score,epoch,optimizer)
            else:
                self.counter +=1
                self.logger.debug(f"[EarlyStopping] No improvement ({self.counter}/{self.patience})")
                

        elif self.mode == 'max':
            if score > (self.best_score+self.delta):
                self.counter = 0
                self.best_score = score
                self.logger.info(f"[EarlyStopping] best score updated to {score:.5f}")
                self.save_checkpoint(model,score,self.best_score,epoch,optimizer)
            else:
                self.counter +=1
                self.logger.debug(f"[EarlyStopping] No improvement ({self.counter}/{self.patience})")
        if self.counter >=self.patience:
            self.early_stop = True
            self.logger.warning(f"[EarlyStopping] Triggered! Patience {self.patience} exceeded. Best score: {self.best_score:.5f}")
        else:
            self.early_stop = False

    def save_checkpoint(self,model,score,best_score,epoch,optimizer):
        self.logger.info(f'Score improved ({best_score:.6f} → {score:.6f}). Saving model ...')
        torch.save({
            'epoch': epoch,
            'model_state_dict': model.state_dict(),
            'optimizer_state_dict': optimizer.state_dict(),
            'loss/acc': best_score,
            }, self.path)
        
        
    @staticmethod
    def load_checkpoint(path, model, optimizer=None):
        checkpoint = torch.load(path)
        model.load_state_dict(checkpoint['model_state_dict'])
        if optimizer is not None:
            optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
        epoch = checkpoint['epoch']
        score = checkpoint['loss/acc']
        return model, optimizer, epoch, score