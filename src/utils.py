import logging
from torch.optim.lr_scheduler import ReduceLROnPlateau
from src.early_stopping import EarlyStopping
from torch.optim.lr_scheduler import ReduceLROnPlateau
from torch.utils.tensorboard import SummaryWriter
import mlflow
from src.logger_utils import setup_logger
import torch
from src.model import SimpleCNN
import argparse
import json


def parse_args():
    parser = argparse.ArgumentParser(description="Digit Recognizer 학습 설정")
    parser.add_argument("--config",type = str, help = "Json config 파일경로")
    parser.add_argument("--train_path",type = str,help = "Train_data 경로")
    parser.add_argument("--test_path",type = str,help = "Test_data 경로")
    parser.add_argument("--lr",type = float,default= 0.001,help = "lr")
    parser.add_argument("--epochs",type = int,default= 50,help = "epochs")
    parser.add_argument("--batch_size",type = int,default= 64,help = "batch_size")
    parser.add_argument('--optimizer', type=str, default='adam',help="optimizer")
    parser.add_argument("--weight_decay",type = float,default= 0.001,help = "weight_decay")
    parser.add_argument("--logging_path",type = str,help = "logging_path 경로")
    parser.add_argument("--logging_basic_level",type = str, default='DEBUG',help = "logging_basic_level, EX : DEBUG, INFO , ERROR , WARNING ")
    parser.add_argument("--logging_console_level",type = str,default='INFO',help = "logging_console_level, EX : DEBUG, INFO , ERROR , WARNING")
    parser.add_argument('--logging_file_level', type=str, default='INFO',help="logging_file_level , EX : DEBUG, INFO , ERROR , WARNING")
    parser.add_argument("--mlflow_log_path",type = str,help = "Mlflow로그 저장 경로")
    parser.add_argument("--tensorboard_log_path",type = str,help = "Tensorboard로그 저장 경로")
    parser.add_argument('--scheduler_patience', type=int, default=3,help = "스케쥴러 개선 횟수 ")
    parser.add_argument('--scheduler_lr_factor', type=float, default=0.5,help = "스케쥴러 lr줄이는 비율")
    parser.add_argument('--scheduler_mode', type=str, default='min',choices=['min', 'max'],help = "스케쥴러 모드 ")
    parser.add_argument('--earlystop_patience', type=int, default=3, help="EarlyStopping: 개선 없을 때 기다릴 에폭 수")
    parser.add_argument('--earlystop_delta', type=float, default=0.001, help="EarlyStopping: 개선으로 인정할 최소 변화량")
    parser.add_argument('--earlystop_mode', type=str, default='min', choices=['min', 'max'], help="EarlyStopping: loss or metric 기준 모드")
    
    args = parser.parse_args()

    # config 파일로 덮어쓰기
    if args.config:
        with open(args.config, 'r') as f:
            config_dict = json.load(f)
        for key, value in config_dict.items():
            if getattr(args, key) == parser.get_default(key): # 중복방지
                setattr(args, key, value)
    return args


LOG_LEVELS = {
    'CRITICAL': logging.CRITICAL,
    'ERROR': logging.ERROR,
    'WARNING': logging.WARNING,
    'INFO': logging.INFO,
    'DEBUG': logging.DEBUG,
    'NOTSET': logging.NOTSET
}
OPTIMIZER_FACTORY = {
    'adam': torch.optim.Adam,
    'sgd': torch.optim.SGD,
    'adamw': torch.optim.AdamW,
    'rmsprop': torch.optim.RMSprop
}


def calculate_accuracy(preds, labels):
    logger = logging.getLogger(__name__)
    predicted = preds.argmax(dim=1) #가장 높은 점수를 받은 클래스의 인덱스를 뽑기
    correct = (predicted == labels).sum().item() # 예측값과 실제값이 일치하는 개수를 세기
    # logger.debug(f"calculate_accuracy_result : {correct}")
    return correct

def get_gradient_snapshot(model, exclude_bias=True, reduction='mean'):
    logger = logging.getLogger(__name__)
    """
    모델의 gradient snapshot 추출.

    Args:
        model (torch.nn.Module): 모델 객체
        exclude_bias (bool): bias 파라미터 제외 여부
        reduction (str): 'mean' 또는 'max'

    Returns:
        List[float]: 각 layer의 gradient 크기 리스트
    """
    snapshot = []
    for name,p in model.named_parameters():
        if p.requires_grad and (not exclude_bias or 'bias' not in name) and p.grad is not None:
            if reduction == 'mean':
                snapshot.append(p.grad.abs().mean().item())
            elif reduction == 'max':
                snapshot.append(p.grad.abs().max().item())
            else:
                logger.warnging("reduction must be 'mean' or 'max'")
                raise ValueError("reduction must be 'mean' or 'max'")
    return snapshot


def setup_tensorboard(args):
    return SummaryWriter(log_dir=args.tensorboard_log_path)

def setup_mlflow(args):
    mlflow.set_tracking_uri(args.mlflow_log_path)
    mlflow.set_experiment("digit-recognizer")


def setup_logging(args):
    logging.getLogger("PIL").setLevel(logging.WARNING)
    logging.getLogger("matplotlib").setLevel(logging.WARNING)
    level = LOG_LEVELS.get(args.logging_basic_level.upper(), logging.DEBUG)
    console_level = LOG_LEVELS.get(args.logging_console_level.upper(), logging.INFO)
    file_level = LOG_LEVELS.get(args.logging_file_level.upper(), logging.INFO)
    setup_logger(file_path=args.logging_path,level= level, console_level=console_level, file_level=file_level)

def prepare_model():
    model = SimpleCNN()
    model.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    return model.to(model.device)

def create_optimizer(model, args):
    opt_cls = OPTIMIZER_FACTORY.get(args.optimizer.lower())
    if opt_cls is None:
        raise ValueError(f"Unknown optimizer: {args.optimizer}")
    return opt_cls(model.parameters(), lr=args.lr,weight_decay = args.weight_decay)


def prepare_training_tools(optimizer, args):
    criterion = torch.nn.CrossEntropyLoss()
    scheduler = ReduceLROnPlateau(
        optimizer,
        mode=args.scheduler_mode,
        patience=args.scheduler_patience,
        factor=args.scheduler_lr_factor
    )
    early_stopper = EarlyStopping(patience=args.earlystop_patience, delta=args.earlystop_delta, mode=args.earlystop_mode)
    return criterion, scheduler, early_stopper