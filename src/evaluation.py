import time
import torch
from util import get_device


def test_with_dataloader_points(model, dataloader, inside):
    start_time = time.time()

    model.eval()

    device = get_device()
    model = model.to(device)

    old_batch_size = dataloader.batch_size
    dataloader.batch_size = 1_000_000

    true_estimations = 0
    num_points = 0
    for points, _ in dataloader:
        points = points.to(device)
        with torch.no_grad():
            output = model(points)

        if inside:
            correct = output <= 0.0
        else:
            correct = output > 0.0
        true_estimations += torch.sum(correct).item()
        num_points += len(points)
    
    percent_right = true_estimations / num_points * 100

    test_time = time.time() - start_time

    num_wrongly_classified_points = num_points - true_estimations

    dataloader.batch_size = old_batch_size

    return percent_right, num_wrongly_classified_points, test_time