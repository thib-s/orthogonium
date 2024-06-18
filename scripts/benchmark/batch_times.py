import functools
import itertools
import time

import numpy as np
import torch

LR = 1e-3


def add_unit(time_in_seconds: float):
    if time_in_seconds < 0:
        return "-" + add_unit(-time_in_seconds)
    if time_in_seconds < 1:
        return f"{time_in_seconds * 1000:.2f} ms"
    if time_in_seconds > 60**2:
        return f"{time_in_seconds / 60**2:.2f} h"
    if time_in_seconds > 60:
        return f"{time_in_seconds / 60:.2f} min"
    return f"{time_in_seconds:.2f} sec"


def evaluate_seconds_per_batch(evaluation_function, loader, nrof_batches=1):
    batch_times = []
    for i, (batch, _) in enumerate(itertools.cycle(loader)):
        if i >= nrof_batches:
            break
        start = time.time()
        evaluation_function(batch)
        end = time.time()
        batch_times.append(end - start)
    return batch_times


@torch.no_grad()
def forward_pass(model, batch, device="cuda"):
    batch = batch.to(device)
    out = model(batch)
    out.sum()


def backward_pass(model, optimizer, batch, device="cuda"):
    batch = batch.to(device)
    out = model(batch)
    out.sum().backward()
    optimizer.step()


def evaluate_seconds_per_training_batch(
    model, train_loader, nrof_batches=1, device="cuda"
):
    model.train()
    model.to(device)
    optimizer = torch.optim.SGD(model.parameters(), lr=LR, momentum=0.9)
    model_backward = functools.partial(backward_pass, model, optimizer, device=device)
    batch_times = evaluate_seconds_per_batch(
        model_backward, train_loader, nrof_batches=nrof_batches
    )
    return batch_times


def evaluate_second_per_validation_batch(
    model, val_loader, nrof_batches=1, cached=True, device="cuda"
):
    model.eval()
    model.to(device)
    model_forward = functools.partial(forward_pass, model, device=device)

    evaluation_function = evaluate_seconds_per_batch
    if cached:
        evaluation_function = torch.nn.utils.parametrize.cached()(evaluation_function)

    val_batch_times = evaluation_function(
        model_forward, val_loader, nrof_batches=nrof_batches
    )
    return val_batch_times


def evaluate_all_model_time_statistics(
    model: torch.nn.Module,
    train_loader,
    test_loader,
    nrof_batches: int = 100,
    log=None,
):
    if log is None:

        def log(*args, **kwargs):
            pass

    device = "cuda" if torch.cuda.is_available() else "cpu"
    log(f'Using device "{device}".')
    log(f"\n\nGot following measurements:")

    train_batch_times = evaluate_seconds_per_training_batch(
        model, train_loader, nrof_batches=1 + nrof_batches, device=device
    )
    train_time_str = ", ".join([add_unit(t) for t in train_batch_times[:5]])
    log("Train batch times:", train_time_str, "...")
    train_mean = np.mean(train_batch_times[1:])
    train_std = np.std(train_batch_times[1:])
    log(f"Train mean: {add_unit(train_mean)}, std: {add_unit(train_std)}")

    test_batch_times = evaluate_second_per_validation_batch(
        model, test_loader, nrof_batches=1 + nrof_batches, device=device
    )
    test_time_str = ", ".join([add_unit(t) for t in test_batch_times[:5]])
    log("Test batch times:", test_time_str, "...")

    # Track more actually:
    test_cached_mean = np.mean(test_batch_times[1:])
    test_cached_std = np.std(test_batch_times[1:])
    log(
        f"Test cached mean: {add_unit(test_cached_mean)}, "
        f"std: {add_unit(test_cached_std)}"
    )

    test_caching_time = test_batch_times[0] - test_cached_mean
    log(f"Caching Time: {add_unit(test_caching_time)}")

    return {
        "train_mean": train_mean,
        "train_std": train_std,
        "test_cached_mean": test_cached_mean,
        "test_cached_std": test_cached_std,
        "test_caching_time": test_caching_time,
    }
