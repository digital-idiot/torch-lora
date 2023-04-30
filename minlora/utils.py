from minlora import LoRAParametrization
from torch import nn
from typing import Literal, Callable, Optional


def apply_to_lora(fn) -> Callable:
    """apply a function to LoRAParametrization layers, designed to be used with model.apply"""

    def apply_fn(layer):
        if isinstance(layer, LoRAParametrization):
            fn(layer)

    return apply_fn


enable_lora = lambda model: model.apply(apply_to_lora(lambda x: x.enable_lora()))
disable_lora = lambda model: model.apply(apply_to_lora(lambda x: x.disable_lora()))


# ------------------- helper function for collecting parameters for training/saving -------------------


def if_lora(name: str) -> bool:
    parts = name.split(".", -1)
    return (
        len(parts) >= 4
        and (parts[-4]) == "parametrizations"
        and parts[-1] in ["lora_A", "lora_B"]
    )


def if_bias(name: str) -> bool:
    return name.rsplit(".", 1)[-1] == "bias"


def if_any(name: str) -> bool:
    parts = name.split(".", -1)
    return (
        len(parts) >= 4
        and (parts[-4]) == "parametrizations"
        and parts[-1] in ["lora_A", "lora_B"]
    ) or (parts[-1] == "bias")

filter_bank = {"lora": if_lora, "bias": if_bias, "both": if_any}


def filter_params(
    model: nn.Module,
    print_shapes: Optional[bool] = False,
    named: Optional[bool] = False,
    name_filter: Optional[Callable] = None
):
    for n, p in model.named_parameters(
        prefix='',
        recurse=True,
        remove_duplicate=True
    ):
        if name_filter is None or name_filter(n):
            if print_shapes:
                print(n, p.shape)
            yield (n, p) if named else p

def parameter_groups(model: nn.Module):
    omit_params = list()
    lora_params = list()
    bias_params = list()
    for n, p in model.named_parameters(
        prefix='',
        recurse=True,
        remove_duplicate=True
    ):
        if if_lora(n):
            lora_params.append(p)
        elif if_bias(n):
            bias_params.append(p)
        else:
            omit_params.append(p)
    return {
        "lora": lora_params,
        "bias": bias_params,
        "omit": omit_params
    }

def get_parameters(
    model: nn.Module,
    named Optional[bool] = False,
    key: Optional[Literal["lora", "bias", "both", None]] = None
):
    return filter_params(
        model=model,
        print_shapes=False,
        named=named,
        name_filter=filter_bank.get(key, None)
    )


def get_lora_state_dict(model):
    return {k: v for k, v in model.state_dict().items() if name_is_lora(k)}


# ------------------- helper function for inferencing with multiple lora -------------------


def _prepare_for_multiple_lora(lora_layer):
    lora_layer.lora_As = list()
    lora_layer.lora_Bs = list()


def _append_lora(lora_layer):
    lora_layer.lora_As.append(nn.Parameter(lora_layer.lora_A.clone()))
    lora_layer.lora_Bs.append(nn.Parameter(lora_layer.lora_B.clone()))


def load_multiple_lora(model, lora_state_dicts):
    model.apply(apply_to_lora(_prepare_for_multiple_lora))
    for state_dict in lora_state_dicts:
        _ = model.load_state_dict(state_dict, strict=False)
        model.apply(apply_to_lora(_append_lora))
    return model


def _select_lora(lora_layer, index):
    lora_layer.lora_A = lora_layer.lora_As[index]
    lora_layer.lora_B = lora_layer.lora_Bs[index]


def select_lora(model, index):
    model.apply(apply_to_lora(lambda x: _select_lora(x, index)))
    return model


# ------------------- helper function for tying and untieing weights -------------------


def tie_weights(linear: nn.Linear, embedding: nn.Embedding):
    """tie the weights of the linear layer and the embedding layer both with the same lora"""
    # this line below is optional if the original is already tied
    embedding.parametrizations.weight.original = linear.parametrizations.weight.original
    embedding.parametrizations.weight[0].lora_A = linear.parametrizations.weight[0].lora_B
    embedding.parametrizations.weight[0].lora_B = linear.parametrizations.weight[0].lora_A


def untie_weights(linear: nn.Linear, embedding: nn.Embedding):
    """untie the weights of the linear layer and the embedding layer"""
    embedding.parametrizations.weight.original = nn.Parameter(embedding.weight.original.clone())
    embedding.parametrizations.weight[0].lora_A = nn.Parameter(embedding.parametrizations.weight[0].lora_A.clone())
    embedding.parametrizations.weight[0].lora_B = nn.Parameter(embedding.parametrizations.weight[0].lora_B.clone())
