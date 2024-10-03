import torch
from torchvision.ops.boxes import box_area


def box_cxcywh_to_xyxy(x):
    x_c, y_c, w, h = x.unbind(-1)
    b = [(x_c - 0.5 * w), (y_c - 0.5 * h),
         (x_c + 0.5 * w), (y_c + 0.5 * h)]
    return torch.stack(b, dim=-1)

def box_cxcywh_to_xywh(x):
    x_c, y_c, w, h = x.unbind(-1)
    b = [(x_c - 0.5 * w), (y_c - 0.5 * h),
         w, h]
    return torch.stack(b, dim=-1)

def box_xywh_to_xyxy(x):
    x1, y1, w, h = x.unbind(-1)
    b = [x1, y1, x1 + w, y1 + h]
    return torch.stack(b, dim=-1)


def box_xyxy_to_xywh(x):
    x1, y1, x2, y2 = x.unbind(-1)
    b = [x1, y1, x2 - x1, y2 - y1]
    return torch.stack(b, dim=-1)


def box_xyxy_to_cxcywh(x):
    x0, y0, x1, y1 = x.unbind(-1)
    b = [(x0 + x1) / 2, (y0 + y1) / 2,
         (x1 - x0), (y1 - y0)]
    return torch.stack(b, dim=-1)


# Modified from torchvision to also return the union
# Note that this function only supports shape (N,4)


def box_iou(boxes1, boxes2):
    """
    Args:
        boxes1: (N, 4) (x1,y1,x2,y2)
        boxes2: (N, 4) (x1,y1,x2,y2)
    """

    area1 = box_area(boxes1)  # (N,)
    area2 = box_area(boxes2)  # (N,)

    lt = torch.max(boxes1[:, :2], boxes2[:, :2])  # (N,2)
    rb = torch.min(boxes1[:, 2:], boxes2[:, 2:])  # (N,2)

    wh = (rb - lt).clamp(min=0)  # (N,2)
    inter = wh[:, 0] * wh[:, 1]  # (N,)

    union = area1 + area2 - inter

    iou = inter / union
    return iou, union

'''Note that this implementation is different from DETR's'''
def visualizeDuringTraining(img_input, Box, isNormed=True, imgSize=384, needRescale=True):
    try:
        img = img_input.cpu().detach().numpy()
    except:
        img = img_input
    if isNormed:
        mean = np.array([0.485, 0.456, 0.406])
        std = np.array([0.229, 0.224, 0.225])
        img[0] = (img[0] * std[0] + mean[0]) * 255
        img[1] = (img[1] * std[1] + mean[1]) * 255
        img[2] = (img[2] * std[2] + mean[2]) * 255
        img = img.transpose(1, 2, 0).astype(np.int8).copy()
    box = Box.cpu().detach().numpy()
    if needRescale:
        box *= imgSize
    left = int(box[0] - 0.5 * box[2])
    top = int(box[1] - 0.5 * box[3])
    width = int(box[2])
    height = int(box[3])
    cv2.rectangle(img, (left, top), (left + width, top + height),
        (0, 0, 255), 1)
    cv2.imwrite("pre.jpg", img)


# Note that this implementation is different from DETR


def generalized_box_iou(boxes1, boxes2):
    """
    Generalized IoU from https://giou.stanford.edu/

    The boxes should be in [x0, y0, x1, y1] format.

    boxes1: (N, 4)
    boxes2: (N, 4)
    """

    # Degenerate boxes gives inf / nan results
    # So do an early check
    # try:
    assert (boxes1[:, 2:] >= boxes1[:, :2]).all()
    assert (boxes2[:, 2:] >= boxes2[:, :2]).all()
    iou, union = box_iou(boxes1, boxes2)  # (N,)

    lt = torch.min(boxes1[:, :2], boxes2[:, :2])
    rb = torch.max(boxes1[:, 2:], boxes2[:, 2:])

    wh = (rb - lt).clamp(min=0)  # (N,2)
    area = wh[:, 0] * wh[:, 1]  # (N,)
    return iou - (area - union) / area, iou


def giou_loss(boxes1, boxes2):
    """
    Args:
        boxes1: (N, 4) (x1,y1,x2,y2)
        boxes2: (N, 4) (x1,y1,x2,y2)
    """

    giou, iou = generalized_box_iou(boxes1, boxes2)
    return (1 - giou).mean(), iou


def clip_box(box: list, H, W, margin=0):
    x1, y1, w, h = box
    x2, y2 = x1 + w, y1 + h
    x1 = min(max(0, x1), W - margin)
    x2 = min(max(margin, x2), W)
    y1 = min(max(0, y1), H - margin)
    y2 = min(max(margin, y2), H)
    w = max(margin, x2 - x1)
    h = max(margin, y2 - y1)
    return [x1, y1, w, h]
