## Understanding Framework

加载 `config`：

`internlm/initialize/launch.py:launch_from_torch`

```python
def launch_from_torch(
    config: Union[str, Path, Config, Dict],
    backend: str = "nccl",
    seed: int = 1024,
):
    """A wrapper for internlm.launch for torchrun or torch.distributed.launch by reading rank and world size
    from the environment variables set by PyTorch

    Args:
        config (Union[str, dict, Config]): Config file or config file path are both acceptable
        backend (str, optional): Backend for ``torch.distributed``, defaults to ``nccl``
        seed (int, optional): Specified random seed for every process. Defaults to 1024.
    """
    try:
        rank = int(os.environ["RANK"])
        local_rank = int(os.environ["LOCAL_RANK"])
        world_size = int(os.environ["WORLD_SIZE"])
        host = os.environ["MASTER_ADDR"]
        port = int(os.environ["MASTER_PORT"])
    except KeyError as e:
        raise RuntimeError(f"Could not find {e} in the torch environment")

    try_bind_numa(global_rank=rank, world_size=world_size, local_rank=local_rank)

    launch(
        config=config,
        local_rank=local_rank,
        rank=rank,
        world_size=world_size,
        host=host,
        port=port,
        backend=backend,
        seed=seed,
    )
```

## 修改

修改 `input_ids` 插入 `CLS`：

位置：`internlm/model/modeling_internlm2.py:472`
函数：`forward`

修改 `label` 插入 `-100`：

位置：`internlm/core/scheduler/no_pipeline_scheduler.py:125`
函数：`_train_one_batch`


