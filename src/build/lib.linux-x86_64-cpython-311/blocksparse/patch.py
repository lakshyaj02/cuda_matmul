import re
import torch

from .blockedell_linear import BlockedEllLinear

class ModelPatcher:
    def __init__(self):
        self.patterns = []

    def is_patchable(self, module_name, module, raiseError):
        return True

    def get_patchable_layers(self, model, num_layers=1):
        # Layer names (displayed as regexps)")
        ret = []
        patch = []
        for k, v in model.named_modules():
            if self.is_patchable(k, v, raiseError=False) :
                if model.config.model_type == "opt" and 'layers' in k:
                    patch.append(k)
                    r = re.escape(k)
                    ret.append({"regexp": r, "layer": v})
                else:
                    patch.append(k)
                    r = re.escape(k)
                    ret.append({"regexp": r, "layer": v})
        return ret, patch

    def add_pattern(self, pattern, patch_info, block_shape=(8,8), block_mask=None):
        self.patterns.append(dict(pattern=pattern, patch_info=patch_info, block_shape=block_shape, block_mask=block_mask))

    def pattern_match(self, module_name):
        for pattern_def in self.patterns:
            if re.match(pattern_def["pattern"], module_name):
                return True, pattern_def["patch_info"], pattern_def["block_shape"], pattern_def["block_mask"]
        return False, -1, (8,8), None

    def new_child_module(self, child_module_name, child_module, patch_info):
        raise NotImplementedError("Implement this in subclasses")

    def replace_module(self, father, child_module_name, child_name, child_module, patch_info, block_shape, block_mask=None):
        new_child_module = self.new_child_module(child_module_name, child_module, patch_info, block_shape, block_mask)
        if new_child_module is not None:
            setattr(father, child_name, new_child_module)

    def patch_model(self, model):
        modules = {}
        modified = False
        for k, v in model.named_modules():
            modules[k] = v
            match, patch_info, block_shape, block_mask = self.pattern_match(k)
            if match and self.is_patchable(k, v, raiseError=True):
                parts = k.split(".")
                father_module_name = ".".join(parts[:-1])
                child_name = parts[-1]
                father = modules[father_module_name]
                self.replace_module(father, k, child_name, v, patch_info, block_shape, block_mask)
                modified = True
        if not modified:
            print(
                "Warning: the patcher did not patch anything!"
                " Check patchable layers with `mp.get_patchable_layers(model)`"
            )

        del modules
        torch.cuda.empty_cache()

class BlockedEllModelPatcher(ModelPatcher):
    """
    Patch a model with block sparse layers
    """

    def is_patchable(self, module_name, module, raiseError):
        if isinstance(module, torch.nn.Linear):
            return True
        else:
            if raiseError:
                raise Exception(f"Cannot patch {module_name}: this is not a Linear layer:\n{module}")
            return False

    def new_child_module(self, child_module_name, child_module, patch_info, block_shape=(8,8), block_mask=None):
        density = patch_info["density"]
        patch_type = "BlockedEllLinear"

        self.is_patchable(child_module_name, child_module, raiseError=True)
        print(
            f"Patching with {patch_type} '{child_module_name}' with density={density}, in={child_module.in_features},"
            f" out={child_module.out_features},bias={child_module.bias is not None} "
        )
        ret = BlockedEllLinear(0, 0, torch_nn_linear=child_module, block_size=block_shape, block_mask=block_mask)
        return ret
