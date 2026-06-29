from aicybops_lib.tracking.mlflow_logging import Logger
import mlflow

def test_logging_basic():
    logger = Logger()
    
    with logger.run("Test Experiment") as run:
        logger.log_params({"param1": 1})
        logger.log_metrics({"metric1": 0.95})
        assert run is not None
        assert logger.has_active_run()

def test_nested_runs():
    logger = Logger()
    
    with logger.run("Parent Experiment") as parent_run:
        logger.log_params({"parent_param": 1})
        
        with logger.run("Child Experiment") as child_run:
            logger.log_metrics({"child_metric": 0.5})
            assert logger.has_active_run()
            assert child_run is not None
        
        assert logger.has_active_run()
        assert parent_run is not None

def test_manual_run_management():
    logger = Logger()
    with logger.run("Manual Experiment") as run:
        assert logger.has_active_run()
        logger.log_params({"test_param": 1})
        logger.log_metrics({"test_metric": 0.75})
    
    assert not logger.has_active_run()

def test_run_stack_management():
    logger = Logger()
    with logger.run("Level 1") as run1:
        assert mlflow.active_run() is not None
        assert logger.has_active_run()
        with logger.run("Level 2") as run2:
            assert mlflow.active_run() is not None
            assert run2 is not None
            with logger.run("Level 3") as run3:
                assert mlflow.active_run() is not None
                assert run3 is not None
            assert mlflow.active_run() is not None
        assert mlflow.active_run() is not None
    assert mlflow.active_run() is None
    assert not logger.has_active_run()
