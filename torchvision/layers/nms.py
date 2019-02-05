from torchvision import _C


def nms(dets, scores, threshold):
    """
    Perform non-maximum suppresion on a set of detection bounding boxes

    Args:
        dets (Tensor): Bounding box detections to perform NMS on. Shape is (N, 4).
        scores (Tensor): Confidence scores for each bounding box detection. Shape is (N,).
        threshold (float): NMS overlap threshold.

    Returns:
        Tensor: Indices of detections which have been preserved.
    """
    return _C.nms(dets, scores, threshold)
