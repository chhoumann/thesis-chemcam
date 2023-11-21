import ray
import typer


app = typer.Typer()


def train_step():
    pass


def val_step():
    pass


def train_loop_per_worker():
    pass


def train_model(experiment_name: str, dataset_loc: str):
    pass


# madewithml/train.py
if __name__ == "__main__":
    if ray.is_initialized():
        ray.shutdown()
    ray.init()
    app()  # initialize Typer app
