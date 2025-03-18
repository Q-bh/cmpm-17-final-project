import wandb
import time
import random

run = wandb.init(project = "example-test", name = "my-run")

for i in range(100):
    run.log({"train loss": i, "validation loss": i + random.random()})
    time.sleep(0.1)