
from typing import Any, Callable, Dict
import torch
from torch import FloatTensor, Tensor, nn
from torch.utils.data import DataLoader
from tqdm.auto import tqdm
from oml.interfaces.datasets import IBaseDataset, IIndexedDataset
from oml.utils.misc_torch import get_device, temporary_setting_model_mode, unique_by_ids

@torch.no_grad()
def _inference(
    model: nn.Module,
    apply_model: Callable[[nn.Module, Dict[str, Any]], Tensor],
    dataset: IIndexedDataset,
    num_workers: int,
    batch_size: int,
    verbose: bool,
    use_fp16: bool,
    accumulate_on_cpu: bool = True,
) -> Tensor:
    loader = DataLoader(dataset=dataset, batch_size=batch_size, num_workers=num_workers, shuffle=False)
    if verbose:
        loader = tqdm(loader, desc=str(get_device(model)))

    outputs_list = []
    ids = []

    with torch.autocast(device_type="cuda", dtype=torch.float16 if use_fp16 else torch.float32):
        with temporary_setting_model_mode(model, set_train=False):
            for batch in loader:
                out = apply_model(model, batch)
                if accumulate_on_cpu:
                    out = out.cpu()
                outputs_list.append(out)
                ids.extend(batch[dataset.index_key].long().tolist())

    outputs = torch.cat(outputs_list).detach()
    ids, outputs = unique_by_ids(ids=ids, data=outputs)

    assert len(outputs) == len(dataset), "Data was not collected correctly after DDP sync."
    assert list(range(len(dataset))) == ids, "Data was not collected correctly after DDP sync."

    return outputs

@torch.no_grad()
def inference(
    model: nn.Module,
    dataset: IBaseDataset,
    batch_size: int,
    num_workers: int = 0,
    verbose: bool = False,
    use_fp16: bool = False,
    accumulate_on_cpu: bool = True,
) -> Tensor:
    device = get_device(model)

    def apply(model_: nn.Module, batch_: Dict[str, Any]) -> FloatTensor:
        return model_(batch_[dataset.input_tensors_key].to(device))

    return _inference(
        model=model,
        apply_model=apply,
        dataset=dataset,
        num_workers=num_workers,
        batch_size=batch_size,
        verbose=verbose,
        use_fp16=use_fp16,
        accumulate_on_cpu=accumulate_on_cpu,
    )