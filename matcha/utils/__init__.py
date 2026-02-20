from matcha.utils.pylogger import get_pylogger

def __getattr__(name):
    if name in ("instantiate_callbacks", "instantiate_loggers"):
        from matcha.utils.instantiators import instantiate_callbacks, instantiate_loggers
        return {"instantiate_callbacks": instantiate_callbacks, "instantiate_loggers": instantiate_loggers}[name]
    if name == "log_hyperparameters":
        from matcha.utils.logging_utils import log_hyperparameters
        return log_hyperparameters
    if name in ("enforce_tags", "print_config_tree"):
        from matcha.utils.rich_utils import enforce_tags, print_config_tree
        return {"enforce_tags": enforce_tags, "print_config_tree": print_config_tree}[name]
    if name in ("extras", "get_metric_value", "task_wrapper"):
        from matcha.utils.utils import extras, get_metric_value, task_wrapper
        return {"extras": extras, "get_metric_value": get_metric_value, "task_wrapper": task_wrapper}[name]
    raise AttributeError(f"module 'matcha.utils' has no attribute {name!r}")
