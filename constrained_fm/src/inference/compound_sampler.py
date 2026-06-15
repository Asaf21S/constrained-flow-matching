import torch


def generate_compound_constrained_points(num_points: int, boxes: torch.Tensor, predictor: torch.nn.Module,
                                         generator: torch.nn.Module, device=None):
    """
    Pipeline to generate points across multiple disjoint bounding boxes using
    amortized importance sampling (AreaPredictor).
    """
    K = boxes.shape[0]

    predictor.eval()
    with torch.no_grad():
        raw_masses = predictor(boxes).squeeze(-1)

    if raw_masses.sum() <= 1e-8:
        print("Warning: Predictor output 0 for all boxes. Falling back to uniform distribution.")
        probs = torch.ones_like(raw_masses) / K
    else:
        probs = raw_masses / raw_masses.sum()

    print("Predicted Normalized Probabilities:", probs.cpu().numpy())

    box_assignments = torch.multinomial(probs, num_samples=num_points, replacement=True)
    points_per_box = torch.bincount(box_assignments, minlength=K)


    all_generated_points = []

    print(f"Generating {num_points} total points across {K} boxes...")
    for i in range(K):
        n_i = points_per_box[i].item()
        if n_i == 0:
            continue

        box = boxes[i].tolist()

        points_i = generator.sample(num_points=n_i, bounds=box, device=device)

        all_generated_points.append(points_i)

    final_point_cloud = torch.cat(all_generated_points, dim=0)

    return final_point_cloud, points_per_box


def generate_from_tiles(num_points, active_boxes, probs, generator, device=None):
    K = active_boxes.shape[0]
    box_assignments = torch.multinomial(probs, num_samples=num_points, replacement=True)
    points_per_box = torch.bincount(box_assignments, minlength=K)

    all_generated_points = []
    print(f"Generating {num_points} total points across {K} active tiles...")

    for i in range(K):
        n_i = points_per_box[i].item()
        if n_i == 0: continue

        box = active_boxes[i].tolist()
        points_i = generator.sample(num_points=n_i, bounds=box, device=device)
        all_generated_points.append(points_i)

    return torch.cat(all_generated_points, dim=0)
