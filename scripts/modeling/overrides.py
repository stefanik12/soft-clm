import logging
from typing import NamedTuple, Iterable, Any, Type, Dict

import torch

logger = logging.getLogger()


class ModuleInfo(NamedTuple):
    attr: str
    path: str
    module: torch.nn.Module
    parent: torch.nn.Module


def all_module_parent_pairs(model: torch.nn.Module) -> Iterable[ModuleInfo]:
    for path, module in model.named_modules():
        if path == "":
            assert module is model
            yield ModuleInfo("", "", model, parent=None)
            continue

        *parent_path, attr = path.split(".")
        parent = model
        for child in parent_path:
            parent = getattr(parent, child)
        yield ModuleInfo(attr, path, module, parent)


def patch_modules_with_overrides(model: torch.nn.Module,
                                 orig_module_cls: Type[torch.nn.Module],
                                 new_module_cls: Type[torch.nn.Module],
                                 new_module_params: Dict[str, Any] = ()) -> torch.nn.Module:
    modules = list(all_module_parent_pairs(model))

    replaced = 0
    for module_info in modules:
        module = module_info.module

        if isinstance(module, orig_module_cls):
            module.__class__ = new_module_cls
            for name, value in new_module_params.items():
                setattr(module, name, value)
            replaced += 1

    logger.info("Replaced %s modules of type %s with %s." % (replaced, orig_module_cls, new_module_cls))
    return model
