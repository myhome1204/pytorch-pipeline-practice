import sys
import os
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "..")))
from src.tune.tune_utils import get_asha_scheduler
from src.tune.train_fn import train_cifar
from ray import tune
from functools import partial
import datetime
import os
from src.tune.tune_utils import tune_parse_args,load_yaml_config,setup_logging,get_asha_scheduler,convert_config_dict_to_tune_space,save_best_trial_results,trial_name

def main():
    args = tune_parse_args()
    setup_logging(args)

    scheduler  = get_asha_scheduler(max_num_epochs = args.max_num_epochs,grace_period=args.min_num_epochs)
    raw_confg = load_yaml_config(args.config_file)
    config  = convert_config_dict_to_tune_space(raw_confg)
    timestamp = datetime.datetime.now().strftime("%Y-%m-%d_%H-%M")
    if args.name:
        args.name = f"{args.name}_{timestamp}"

    result = tune.run(
        partial(train_cifar, data_path=args.train_data_path, device="cuda"),
        resources_per_trial={"cpu": 2, "gpu": args.gpus_per_trial},
        config=config,
        num_samples=args.num_samples,
        scheduler=scheduler,
        storage_path=args.storage_path,
        name = args.name,
        # name="ray_tune_exp2",  # 실험 이름
        log_to_file=True,
        trial_name_creator=lambda trial: f"trial_{trial.trial_id[:4]}",  
        # trial_name_creator = trial_name,
        metric="loss",
        mode="min"        
    )
    save_best_trial_results(result,args)

if __name__ == "__main__":
    # Git Bash 사용
    # scripts/tune/train_exp1_lr.sh
    # scripts/tune/train_exp1_full.sh
    # tensorboard 시각화
    # tensorboard --logdir=outputs/hparam
    # tensorboard --logdir=C:/Users/myhom/study/Digit_Recognizer/src/tune/outputs/hparam/run
    main() 


