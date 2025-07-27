import logging
import yaml
import json
import argparse
from ray import tune
from src.logger_utils import setup_logger
from ray.tune.schedulers import ASHAScheduler, FIFOScheduler
import os
from datetime import datetime
from torch.utils.tensorboard import SummaryWriter
from pathlib import Path
from ray.tune.analysis import ExperimentAnalysis


LOG_LEVELS = {
    'CRITICAL': logging.CRITICAL,
    'ERROR': logging.ERROR,
    'WARNING': logging.WARNING,
    'INFO': logging.INFO,
    'DEBUG': logging.DEBUG,
    'NOTSET': logging.NOTSET
}


def setup_tensorboard(args):
    return SummaryWriter(log_dir=args.tensorboard_log_path)


def tune_parse_args():
    parser = argparse.ArgumentParser(description="Hyperparameter 설정")
    parser.add_argument("--name", type = str,default = "",help = "실험 파일 명")
    parser.add_argument("--config_file", type = str,help = "config 파일(yaml) 경로 ")
    parser.add_argument("--num_samples", type = int ,default = 10, help = "hparms 탐색시 최대 조합 수")
    parser.add_argument("--max_num_epochs", type = int ,default = 10, help = "hparms 탐색시 최대 epoch 수")
    parser.add_argument("--min_num_epochs", type = int ,default = 3, help = "hparms 탐색시 최소 epoch 수")
    parser.add_argument("--train_data_path", type = str , help = "data 경로")
    parser.add_argument("--gpus_per_trial", type = int ,default = 11, help = "hparms 탐색시 gpu 사용 수")
    parser.add_argument("--storage_path", type = str ,default = 10, help = "탐색 결과 저장 경로")
    parser.add_argument("--logging_path",type = str,default='DEBUG',help = "logging_path 경로")
    parser.add_argument("--logging_basic_level",type = str,default='DEBUG',help = "logging_basic_level, EX : DEBUG, INFO , ERROR , WARNING ")
    parser.add_argument("--logging_console_level",type = str,default='INFO',help = "logging_console_level, EX : DEBUG, INFO , ERROR , WARNING")
    parser.add_argument('--logging_file_level', type=str, default='INFO',help="logging_file_level , EX : DEBUG, INFO , ERROR , WARNING")
    parser.add_argument('--save_path', type=str,help="최선의 결과 저장 경로")

    return parser.parse_args() 


def convert_config_dict_to_tune_space(config_dict):
    logger = logging.getLogger(__name__)
    converted = {}
    for key, spec in config_dict.items():
        space = spec.get("space")
        if space == "loguniform":
            converted[key] = tune.loguniform(spec["min"], spec["max"])
        elif space == "uniform":
            converted[key] = tune.uniform(spec["min"], spec["max"])
        elif space == "choice":
            converted[key] = tune.choice(spec["values"])
        else:
            logger.debug(f"지원하지 않는 space 유형: {space}")
            raise ValueError(f"지원하지 않는 space 유형: {space}")
        
    logger.info(f"converted : {converted}")

    return converted



def setup_logging(args):
    logging.getLogger("PIL").setLevel(logging.WARNING)
    logging.getLogger("matplotlib").setLevel(logging.WARNING)
    level = LOG_LEVELS.get(args.logging_basic_level.upper(), logging.DEBUG)
    console_level = LOG_LEVELS.get(args.logging_console_level.upper(), logging.INFO)
    file_level = LOG_LEVELS.get(args.logging_file_level.upper(), logging.INFO)
    setup_logger(file_path=args.logging_path,level= level, console_level=console_level, file_level=file_level)




def get_asha_scheduler(max_num_epochs,grace_period):
    return ASHAScheduler(
        # metric="loss",
        # mode="min",
        max_t=max_num_epochs,
        grace_period=grace_period,
        reduction_factor=2,
    )

def get_fifo_scheduler():
    return FIFOScheduler()

def load_yaml_config(path):
    with open(path, 'r') as f:
        return yaml.safe_load(f)
    

def save_best_trial_results(result,args):

    logger = logging.getLogger(__name__)
    os.makedirs(args.save_path, exist_ok=True)
    
    best_trial = result.get_best_trial("loss", "min", "last")
    best_config = best_trial.config
    best_result = {
        "loss": best_trial.last_result["loss"],
        "accuracy": best_trial.last_result["accuracy"],
        "config": best_config,
    }
    logger.info(f"Best trial config: {best_trial.config}")
    logger.info(f"Best trial final validation loss: {best_trial.last_result['loss']}")
    logger.info(f"Best trial final validation accuracy: {best_trial.last_result['accuracy']}")
    timestamp = datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
    config_filename = Path(args.save_path).name
    result_path = os.path.join(args.save_path, f"best_result_{config_filename}_{timestamp}.json")
    # 저장
    with open(result_path, "w") as f:
        json.dump(best_result, f, indent=4)
    logger.info(f"[✔] Best result saved at: {result_path}")
    # print(f"[✔] Best result saved at: {result_path}")



def save_best_trial_results(result, args):
    logger = logging.getLogger(__name__)
    os.makedirs(args.save_path, exist_ok=True)

    best_trial = result.get_best_trial("loss", "min", "last")
    best_config = best_trial.config


    best_result = {
        "loss": best_trial.last_result["loss"],
        "accuracy": best_trial.last_result["accuracy"],
        "config": best_config,
    }

    timestamp = datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
    config_filename = Path(args.config_file).name
    result_path = os.path.join(args.save_path, f"best_result_{config_filename}_{timestamp}.json")

    with open(result_path, "w") as f:
        json.dump(best_result, f, indent=4)

    logger.info(f"Best result saved at: {result_path}")


def trial_name(trial):
    cfg = trial.config
    # 아주 간단하게, trial 이름 요약
    return f"bs{cfg['batch_size']}_lr{cfg['lr']:.4f}"[:40] 