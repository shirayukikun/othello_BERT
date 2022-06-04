import argparse
from pathlib import Path
from typing import List, Dict, Union, Optional
import os
import json
import math
from datetime import datetime

import wandb
from logzero import logger
import _jsonnet as jsonnet
from ray import tune
from ray.tune import CLIReporter
from ray.tune.schedulers import ASHAScheduler, PopulationBasedTraining
from ray.tune.integration.wandb import WandbLoggerCallback
from ray.tune.suggest.optuna import OptunaSearch

from dentaku_tuner import DentakuTuner



def convert_tune_choice_config(config, args):
    assert type(config) is dict, "\"config\" is not dict..."
    output_config = {}
    mutation_hypera_name = set()
    
    # Convert tune choice
    for k, v in config.items():
        if type(v) is dict:
            keys = list(v.keys())
            values = list(v.values())
            assert all(type(l) is list for l in values), \
                "Not all element's are list"
            assert all(len(l) == len(values[0]) for l in values), \
                "Not all list's lengthes are same"
            assert all(hasattr(args, key) for key in keys), \
                f"args dose not have \"{v}\" attribute..."
            
            output_config[k] = tune.choice(
                [
                    {
                        k_t: v_t for k_t, v_t in zip(v.keys(), V_T)
                    }
                    for V_T in zip(*values)
                ]
            )

            for k_t in output_config[k][0].keys():
                mutation_hypera_name.add(k_t)
                            
        elif type(v) is list:
            assert hasattr(args, k), f"args dose not have \"{k}\" attribute..."
            output_config[k] = tune.choice(v)
            mutation_hypera_name.add(k)

        else:
            raise NotImplementedError()

    return output_config, mutation_hypera_name



def parse_tune_space_config(file_path:Path, args):
    tune_space_config = json.loads(
        jsonnet.evaluate_file(
            str(file_path.expanduser()),
            ext_vars=dict(os.environ),
        )
    )

    assert "initial_space" in tune_space_config, "Specifi initial_space!"
    assert type(tune_space_config["initial_space"]) is dict, "Specifi initial_space!"
    assert "during_training_space" in tune_space_config, "Specifi during_training_space!"
    assert type(tune_space_config["during_training_space"]) is dict, "Specifi during_training_space!"

    initial_tune_space, initial_mutation_hypera_name = convert_tune_choice_config(
        tune_space_config["initial_space"],
        args
    )
    tune_space, mutation_hypera_name = convert_tune_choice_config(
        tune_space_config["during_training_space"],
        args
    )

    assert initial_mutation_hypera_name.isdisjoint(mutation_hypera_name),  \
        "Same hypera name are included..."

    for k, v in tune_space.items():
        initial_tune_space[k] = v

    return initial_tune_space, tune_space, list(initial_mutation_hypera_name | mutation_hypera_name)



def exec_tune(tune_config, checkpoint_dir=None, args=None):
    assert not (args.tune_method in ["ASH"]) or checkpoint_dir is None, "checkpoint_dir is used..."
    
    for k, v in tune_config.items():
        if (type(v) is dict) and (not hasattr(args, k)):
            for k_in, v_in in v.items():
                assert hasattr(args, k_in), f"args don't have \"{k}\" attribute..."
                setattr(args, k_in, v_in)
        else:
            assert hasattr(args, k), f"args don't have \"{k}\" attribute..."
            setattr(args, k, v)
    
    tuner = DentakuTuner(args, checkpoint_dir)
    tuner()

    
def main(args):
    start_time = datetime.today().strftime("%Y%m%d%H%M%S")
    ray_tune_dir = args.log_model_output_dir / "ray_tune_dir"
    ray_tune_dir.mkdir(parents=True, exist_ok=True)
    
    initial_tune_space, tune_space, _ = parse_tune_space_config(
        args.tune_space_file_path, args
    )

    
    callbacks = [
        WandbLoggerCallback(
            project=args.project_name,
            group=f"{args.version}_{start_time}",
            log_config=True,
            dir=str(args.log_dir),
        )
    ]
    
    # training_iteration : The number of times tune.report() has been called

    if args.tune_method == "PBT":
        raise NotImplementedError(
            "pytorch lightningのチェックポイントのロードの仕様のため，optimizerのlrが更新されないという問題があった．単純には解決不可能"
        )
        tune_scheduler = PopulationBasedTraining(
            time_attr="training_iteration",
            perturbation_interval=args.perturbation_interval,
            quantile_fraction=args.quantile_fraction,
            resample_probability=args.resample_probability,
            hyperparam_mutations=tune_space,
            log_config=True
        )
        tune_algorithm = None
        
    elif args.tune_method == "ASH":
        assert len(tune_space) == 0, "ASHAScheduler cannot change hypera during training. Donot specifi \"during_training_space\""
        tune_scheduler = ASHAScheduler(
            time_attr="training_iteration",
            max_t=args.tune_num_max_report,
            grace_period=args.grace_period,
            reduction_factor=args.reduction_factor,
        )
        tune_algorithm = None


    elif args.tune_method == "Optuna":
        default_hypera = {}
        for k, v in initial_tune_space.items():
            try:
                default_hypera[k] = getattr(args, k)
            except AttributeError:
                default_hypera[k] = {
                    k_t: getattr(args, k_t)
                    for k_t in v.sample().keys()
                }
            
        print(default_hypera)
        
        tune_algorithm = OptunaSearch(
            seed=args.seed,
            points_to_evaluate=[
                default_hypera
            ],
        )
        tune_scheduler = None
            
        
    else:
         raise NotImplementedError()


    reporter = CLIReporter(
        parameter_columns=list(initial_tune_space.keys()),
        metric_columns=[
            "valid_loss",
            "valid_accuracy",
            "training_iteration",
        ]
    )
    
    train_fn_with_parameters = tune.with_parameters(
        exec_tune,
        args=args,
    )

    
    assert args.gpus_per_trial > 0, "Use gpu!"
    resources_per_trial = {
        "cpu": args.cpus_per_trial,
        "gpu": args.gpus_per_trial,
    }

    
    analysis = tune.run(
        train_fn_with_parameters,
        resources_per_trial=resources_per_trial,
        metric="valid_loss",
        mode="min",
        config=initial_tune_space,
        num_samples=args.tune_num_samples,
        scheduler=tune_scheduler,
        search_alg=tune_algorithm,
        progress_reporter=reporter,
        name=args.tag,
        local_dir=ray_tune_dir,
        callbacks=callbacks,
        keep_checkpoints_num=1,
    )



    
    print("Best hyperparameters found were:\n", analysis.best_config)
    with (ray_tune_dir / "best_config.json").open(mode="w") as f:
        json.dump(analysis.best_config, f)

    wandb.save(ray_tune_dir / "best_config.json")
    logger.info("Tuning finish!")



if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser = DentakuTuner.add_args(parser)
    args = parser.parse_args()
    main(args)
