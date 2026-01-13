from os.path import exists,join
from os import mkdir, scandir
import fnmatch
from typing import Tuple
import warnings
from omegaconf import OmegaConf
import logging
import itertools


def find_yml_files(dir:str)-> Tuple[dict]:
    """
    Look for configuration and sweep file in a directory (dir)

    Args:
        dir (str): path to look for file, usually experiment folder

    Raises:
        FileNotFoundError: dir doesn't contain the config file
        Warning: dir doesn't contain the sweep filed

    Returns:
        Tuple[dict]: config_dict, sweep_dict
    """
    config_control_string = "config"
    sweep_control_string = "sweep"
    
    config, sweep_config = None, None
    
    with scandir(dir) as entries:
    
        for entry in entries:
            if entry.is_file():
                
                if fnmatch.fnmatch(entry.name,f"*{config_control_string}*.yaml"):
                    config = OmegaConf.load(entry.path)
                    
                if fnmatch.fnmatch(entry.name,f"*{sweep_control_string}*.yaml"):
                    sweep_config = OmegaConf.load(entry.path)
                    
    if config is None:
        raise FileNotFoundError("No configuration file found")
    
    if sweep_config is None:
        warnings.warn("No available sweep found")
        config = update_config(config) # update config anyways
    
    return config, sweep_config





def combination_sweep(exp_dir, mode="combination"):
    """
    This function scans for configuration files in a folder and runs accordingly
    to the specified sweep mode:
    
    - "independent": One-at-a-time parameter sweep (like independent_sweep)
    - "combination": Sweep across all possible combinations of parameter values
    
    This function has to be used as decorator on the function "func"
    func (function)-> None: callable function with input (config, save_dir)
    
    It also creates the folder structure for sweeps and passes it to func.
    
    For independent mode, the folder structure looks like:
    
    ./experiments
    |__exp1
    |__exp2 <-- experiment folder "exp_dir"
        |__config.yaml <-- starting config file
        |__config_sweep.yaml <-- sweep instructions
        |__sweeps
            |__sweep_param1
            |__sweep_param2
            |__sweep_param3
                |__sweep_param3_val1
                |__sweep_param3_val2
                    |__config.yaml <-- sweep (modified) config file
                    |__results
    
    For combination mode, the folder structure looks like:
    
    ./experiments
    |__exp1
    |__exp2 <-- experiment folder "exp_dir"
        |__config.yaml <-- starting config file
        |__config_sweep.yaml <-- sweep instructions
        |__combinations
            |__combo_param1_val1_param2_val1
            |__combo_param1_val1_param2_val2
                |__config.yaml <-- sweep (modified) config file
                |__results
                    
    Args:
        exp_dir (str): experiment folder
        mode (str): sweep mode, either "independent" or "combination"
    """
    
    config, sweep_config = find_yml_files(dir=exp_dir)
    # set logging
    logger_info = logging.getLogger("logger_info")
    
    def decorator_repeat(func):
        def wrapper(*args, **kwargs):
        
            if sweep_config is not None:
                if mode == "independent":
                    # Independent sweep logic (like independent_sweep)
                    for cat in sweep_config:
                        for param in sweep_config[cat]:

                            sweep_param_path = join(exp_dir, "sweeps", f"sweep_{param}")
                            if not exists(sweep_param_path):
                                mkdir(sweep_param_path)

                            assert param in config[cat], AssertionError(f"Parameters {param} not in config file!")
                            for val in sweep_config[cat][param]:
                                
                                # logging
                                logger_info.info(f"Sweep at {cat}-{param}: {val}")
                                
                                # update the config file for the current sweep
                                config_sweep = config.copy()
                                config_sweep[cat][param] = val
                                
                                # update the saving dir
                                save_dir = join(sweep_param_path, f"sweep_{param}_{val}")
                                if not exists(save_dir):
                                    mkdir(save_dir)

                                # execute function
                                func(*args, **kwargs, config=config_sweep, save_dir=save_dir)

                                # save config_sweep as yaml file
                                config_sweep_path = join(save_dir, "config.yaml")
                                OmegaConf.save(config_sweep, config_sweep_path)
                
                elif mode == "combination":
                    # Combination sweep logic
                    
                    # Extract all parameters and their values
                    param_values = {}
                    param_categories = {}
                    
                    for cat in sweep_config:
                        for param in sweep_config[cat]:
                            assert param in config[cat], AssertionError(f"Parameters {param} not in config file!")
                            param_values[param] = sweep_config[cat][param]
                            param_categories[param] = cat
                    
                    # Generate all possible combinations
                    param_names = list(param_values.keys())
                    value_combinations = list(itertools.product(*(param_values[param] for param in param_names)))
                    
                    # Create combinations directory
                    combinations_path = join(exp_dir, "combinations")
                    if not exists(combinations_path):
                        mkdir(combinations_path)
                    
                    # Process each combination
                    for combo in value_combinations:
                        # Create a descriptive name for this combination
                        combo_name_parts = []
                        
                        # Update config with this combination
                        config_sweep = config.copy()
                        
                        # Log the combination
                        combo_log_parts = []
                        
                        for i, param in enumerate(param_names):
                            val = combo[i]
                            cat = param_categories[param]
                            
                            # update the config file for the current sweep
                            config_sweep[cat][param] = val
                            config_sweep = update_config(config_sweep)
                            
                            # Add to combo name
                            combo_name_parts.append(f"{param}_{val}")
                            
                            # Add to log
                            combo_log_parts.append(f"{cat}-{param}: {val}")
                        
                        # Log the combination
                        logger_info.info(f"Combination sweep with " + ", ".join(combo_log_parts))
                        
                        # Create directory for this combination
                        combo_name = "combo_" + "_".join(combo_name_parts)
                        combo_dir = join(combinations_path, combo_name)
                        if not exists(combo_dir):
                            mkdir(combo_dir)
                        
                        # Execute function with this combination
                        func(*args, **kwargs, config=config_sweep, save_dir=combo_dir)
                        
                        # Save config
                        config_sweep_path = join(combo_dir, "config.yaml")
                        OmegaConf.save(config_sweep, config_sweep_path)
                
                else:
                    raise ValueError(f"Unknown sweep mode: {mode}. Use 'independent' or 'combination'.")
            
            else:
                func(config, save_dir=exp_dir)   
        return wrapper
    return decorator_repeat












def update_config(config: dict)->dict:
    """
    Updates the config file where placeholders are set 

    Args:
        config (dict): config with placeholders

    Returns:
        dict: updated config
    """    
    
    if config.get("model", {}).get("model_object") == "proT":
    
        if config.model.kwargs.comps_embed_enc == "concat":
            config.model.kwargs.d_model_enc = config.model.embed_dim.enc_val_emb_hidden + config.model.embed_dim.enc_var_emb_hidden
            
        if config.model.kwargs.comps_embed_dec == "concat":    
            config.model.kwargs.d_model_dec = config.model.embed_dim.dec_val_emb_hidden + config.model.embed_dim.dec_var_emb_hidden
            
        if config.model.kwargs.comps_embed_enc == "summation":
            config.model.kwargs.d_model_enc = config.model.embed_dim.d_model_set
            
        if config.model.kwargs.comps_embed_dec == "summation":
            config.model.kwargs.d_model_dec = config.model.embed_dim.d_model_set
            
    return config
