from src.ddp.ddp_metrics_logger import DDPMetricsLogger


def test_singleton():
    logger1 = DDPMetricsLogger()
    logger2 = DDPMetricsLogger()
    assert logger1 is logger2


def test_initialization():
    logger = DDPMetricsLogger(metric1=[1, 2, 3], metric2=4)
    assert logger["metric1"] == [1, 2, 3]
    assert logger["metric2"] == [4]


def test_store():
    logger = DDPMetricsLogger()
    logger.store(metric3=5)
    assert logger["metric3"] == [5]
    logger.store({"metric3": 6, "metric4": 7})
    assert logger["metric3"] == [5, 6]
    assert logger["metric4"] == [7]


def test_log_averages_to_wandb(mocker):
    # Mocking the DistributedWandb class to avoid real logging
    mock_wandb_log = mocker.patch('src.ddp.distributed_wandb.DistributedWandb.log')
    logger = DDPMetricsLogger(metric1=[1, 2, 3])
    logger.log_averages_to_wandb()
    mock_wandb_log.assert_called_with({"Metric1": 2.0}, step=None)


def test_getitem():
    logger = DDPMetricsLogger(metric1=[1, 2, 3])
    assert logger["metric1"] == [1, 2, 3]


def test_synchronize(mocker):
    # Mocking the necessary distributed functions
    mocker.patch('src.ddp.ddp_utils.dist_identity.world_size', new=2)
    mocker.patch('src.ddp.ddp_utils.dist_identity.rank', new=0)
    mocker.patch('torch.distributed.broadcast_object_list', return_value=[{"metric1": [4, 5, 6]}])

    logger = DDPMetricsLogger(metric1=[1, 2, 3])
    logger.synchronize()
    assert logger["metric1"] == [1, 2, 3, 4, 5, 6]

