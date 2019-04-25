import torch
import logging


def set_logger(logger_name, log_file, level=logging.INFO):
    log = logging.getLogger(logger_name)
    formatter = logging.Formatter('%(asctime)s : %(message)s')
    fileHandler = logging.FileHandler(log_file, mode='w')
    fileHandler.setFormatter(formatter)
    streamHandler = logging.StreamHandler()
    streamHandler.setFormatter(formatter)

    log.setLevel(level)
    log.addHandler(fileHandler)
    log.addHandler(streamHandler)


def set_log(args):
    log = {}                                                                                                                                        
    set_logger(
        logger_name=args.log_name, 
        log_file=r'{0}{1}'.format("./logs/", args.log_name))
    log[args.log_name] = logging.getLogger(args.log_name)

    # Log arguments
    for (name, value) in vars(args).items():
        log[args.log_name].info("{}: {}".format(name, value))

    return log


def total_rewards(episodes_rewards, aggregation=torch.mean):
    stacked_rewards = torch.stack(
        [aggregation(torch.sum(rewards, dim=0)) for rewards in episodes_rewards], dim=0)
    rewards = torch.mean(stacked_rewards)
    return rewards.item()
