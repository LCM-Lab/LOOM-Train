from loomtrain.core import (
    LoomSFTModule,
    LoomSFTData,
    LoomDataDict,
    LoomTrainer,
    DeepspeedStrategy,
    DeepspeedConfig,
    SortPackingStrategy,
    DataConfig,
    CheckpointConfig,
    parallel

)


if __name__ == "__main__":
    args = ...

    parallel_config = parallel.ParallelConfig(
        train_batch_size = 2,
        micro_batch_size = 1,
        val_batch_size = 2,
        nnodes = 1,
        devices_per_node = 8,
        cp = 8
    )

    data_config = DataConfig(
        collate_type = 'packing',
        packing_length = 64000,
        val_interval = 20,
        batch_size = 10,
        num_epochs = 1
    )

    train_strategy = DeepspeedStrategy(parallel_config, data_config = data_config)

    data_strategy = SortPackingStrategy(parallel_config, data_config = data_config)

    trainer = LoomTrainer(train_strategy = train_strategy,
                          data_strategy = data_strategy)
    
    module = LoomSFTModule("/path/to/model/")
    
    datamodule = LoomSFTData([
        LoomDataDict(path = "/path/to/data",count = 10, prompt_key = 'prompt', response_key = 'response'),
        ...
    ], max_length = 128000)


    ckpt_cfg = CheckpointConfig(load_dir = '/path/to/load', save_dir = '/path/to/save')

    trainer.fit(
        module = module,
        datamodule = datamodule,
        ckpt_cfg = ckpt_cfg
    )
