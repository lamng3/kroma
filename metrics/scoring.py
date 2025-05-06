from typing import Sequence, Tuple

def calculate_metrics(
    y_true: Sequence[int],
    y_pred: Sequence[int],
    positive_label: int = 1
) -> Tuple[float, float, float]:
    """
    Compute precision, recall, and F1 for binary labels in one pass.
    
    Args:
        y_true: ground‑truth labels
        y_pred: predicted labels
        positive_label: the value considered “positive”
    
    Returns:
        (precision, recall, f1)
    """
    tp = fp = fn = 0
    for yt, yp in zip(y_true, y_pred):
        if yp == positive_label:
            if yt == positive_label:
                tp += 1
            else:
                fp += 1
        else:
            if yt == positive_label:
                fn += 1

    precision = tp / (tp + fp) if (tp + fp) else 0.0
    recall    = tp / (tp + fn) if (tp + fn) else 0.0
    f1        = (2 * precision * recall / (precision + recall)) if (precision + recall) else 0.0

    return precision, recall, f1
