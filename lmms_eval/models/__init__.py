import importlib
import os
import sys
from typing import Literal

from loguru import logger

logger.remove()
log_format = "<green>{time:YYYY-MM-DD HH:mm:ss}</green> | " "<level>{level: <8}</level> | " "<cyan>{name}</cyan>:<cyan>{function}</cyan>:<cyan>{line}</cyan> - " "<level>{message}</level>"
logger.add(sys.stdout, level="WARNING", format=log_format)


AVAILABLE_SIMPLE_MODELS = {
    "bagel": "Bagel",
}

AVAILABLE_CHAT_TEMPLATE_MODELS: dict[str, str] = {}


def get_model(model_name, force_simple: bool = False):
    if model_name not in AVAILABLE_SIMPLE_MODELS and model_name not in AVAILABLE_CHAT_TEMPLATE_MODELS:
        raise ValueError(f"Model {model_name} not found in available models.")

    if model_name in AVAILABLE_CHAT_TEMPLATE_MODELS:
        model_type = "chat"
        AVAILABLE_MODELS = AVAILABLE_CHAT_TEMPLATE_MODELS
    else:
        model_type = "simple"
        AVAILABLE_MODELS = AVAILABLE_SIMPLE_MODELS

    if force_simple and model_name in AVAILABLE_SIMPLE_MODELS:
        model_type = "simple"
        AVAILABLE_MODELS = AVAILABLE_SIMPLE_MODELS

    model_class = AVAILABLE_MODELS[model_name]
    if "." not in model_class:
        model_class = f"lmms_eval.models.{model_type}.{model_name}.{model_class}"

    try:
        model_module, model_class = model_class.rsplit(".", 1)
        module = __import__(model_module, fromlist=[model_class])
        return getattr(module, model_class)
    except Exception as e:
        logger.error(f"Failed to import {model_class} from {model_name}: {e}")
        raise


if os.environ.get("LMMS_EVAL_PLUGINS", None):
    for plugin in os.environ["LMMS_EVAL_PLUGINS"].split(","):
        m = importlib.import_module(f"{plugin}.models")
        for model_name, model_class in getattr(m, "AVAILABLE_MODELS").items():
            AVAILABLE_SIMPLE_MODELS[model_name] = f"{plugin}.models.{model_name}.{model_class}"
