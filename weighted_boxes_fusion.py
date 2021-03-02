import detectron2
import numpy as np
import torch

def bb_intersection_over_union(A, B) -> float:
	xA = max(A[0], B[0])
	yA = max(A[1], B[1])
	xB = min(A[2], B[2])
	yB = min(A[3], B[3])


	# compute the area of intersection rectangle
	interArea = max(0, xB - xA) * max(0, yB - yA)

	if interArea == 0:
			return 0.0

	# compute the area of both the prediction and ground-truth rectangles
	boxAArea = (A[2] - A[0]) * (A[3] - A[1])
	boxBArea = (B[2] - B[0]) * (B[3] - B[1])

	iou = interArea / float(boxAArea + boxBArea - interArea)
	return iou

def filter_boxes(boxes, scores, labels, skip_thr):

	new_boxes = {}
	# t is model number 
	for t in range(len(boxes)):
		# j is index boxes, scores, labels
		for j in range(len(boxes[t])):
			
			score = float(scores[t][j])
			if score < skip_thr:
				continue

			label = int(labels[t][j])
			box_part = boxes[t][j]

			x1 = float(box_part[0])
			y1 = float(box_part[1])
			x2 = float(box_part[2])
			y2 = float(box_part[3])

			if x2 < x1:
				x1, x2 = x2, x1
			if y2 < y1:
				y1, y2 = y2, y1
			if x1 < 0:
				x1 = 0
			if x1 > 1:
				x1 = 1
			if y1 < 0:
				y1 = 0
			if y1 > 1:
				y1 = 1
			if x2 < 0:
				x2 = 0
			if x2 > 1:
				x2 = 1
			if y2 < 0:
				y2 = 0
			if y2 > 1:
				y2 = 1
			if (x2-x1)*(y2-y1)==0:
				continue
        
        			
			b = [int(label), float(score), x1, y1, x2, y2]

			if label not in new_boxes:
				new_boxes[label] = []

			new_boxes[label].append(b)

	# k is labels
	# each labels sort by scores 
	for k in new_boxes:
		current_boxes = np.array(new_boxes[k])
		new_boxes[k] = current_boxes[current_boxes[:, 1].argsort()[::-1]]

	# [label, score, x1, y1, x2, y2]
	return new_boxes

def find_matching_box(boxes_list, new_box, iou_thr):
	best_iou = iou_thr
	best_index = -1
	for i in range(len(boxes_list)):
		box = boxes_list[i]
		# if labels is not same
		if box[0] != new_box[0]:
			continue
		iou = bb_intersection_over_union(box[2:], new_box[2:])
		if iou > best_iou:
			best_index = i
			best_iou = iou

	return best_index, best_iou

def get_weighted_box(boxes):
	box = np.zeros(6, dtype=np.float32)
	conf = 0
	conf_list = []
	for b in boxes:
		box[2:] += (b[1] * b[2:])
		conf += b[1]
		conf_list.append(b[1])
	box[0] = boxes[0][0]
	box[1] = conf / len(conf_list)
	box[2:] /= conf
	return box


def weighted_box_fusion(boxes_list, scores_list, labels_list, skip_thr=0.0, iou_thr=0.55): 
  
  filtered_boxes = filter_boxes(boxes_list, scores_list, labels_list, skip_thr)

  if len(filtered_boxes) == 0:
    return np.zeros((0, 4)), np.zeros((0,)), np.zeros((0,))
  
  overall_boxes = []
  for label in filtered_boxes:
    boxes = filtered_boxes[label]
    new_boxes = []
    weighted_boxes = []

    # Clustering
    for j in range(len(boxes)):
      index, best_iou = find_matching_box(weighted_boxes, boxes[j], iou_thr)

      # it is best score new box
      if index == -1:
        new_boxes.append([boxes[j].copy()])
        weighted_boxes.append(boxes[j].copy())
      else:
        new_boxes[index].append(boxes[j])
        weighted_boxes[index] = get_weighted_box(new_boxes[index])
    
    overall_boxes.append(np.array(weighted_boxes))

  overall_boxes = np.concatenate(overall_boxes, axis=0)
  overall_boxes = overall_boxes[overall_boxes[:, 1].argsort()[::-1]]
  boxes = overall_boxes[:, 2:]
  scores = overall_boxes[:, 1]
  labels = overall_boxes[:, 0]
  return boxes, scores, labels

  
def ensemble(models, weights = None, iou_thr = 0.5, skip_thr = 0.0001):
  # models
  # [model A output, model B output, model C output]

  # preprocessing
  boxes_list = []
  scores_list = []
  labels_list = []
  images_size = None

  assert len(models) > 0

  isize = models[0]['instances'].image_size

  def normalize(boxes_tensor): #x1,y1, x2,y2
    return boxes_tensor / np.array([isize[1], isize[0], isize[1], isize[0]])
  def denormalize(boxes_tensor): #x1,y1, x2,y2
    return boxes_tensor * np.array([isize[1], isize[0], isize[1], isize[0]])


  for i in range(len(models)):
    m_instances = models[i]['instances']
    boxes_list.append(normalize(m_instances.pred_boxes.tensor.numpy()).tolist())
    scores_list.append(m_instances.scores.tolist())
    labels_list.append(m_instances.pred_classes.tolist())

  boxes, scores, labels = weighted_box_fusion(boxes_list, scores_list, labels_list, iou_thr=iou_thr, skip_thr=skip_thr)    
  boxes = denormalize(boxes)

  new_Boxes = detectron2.structures.boxes.Boxes(torch.from_numpy(boxes).to(torch.float32))

  new_fields = {}
  new_fields['pred_boxes'] = new_Boxes
  new_fields['scores'] = torch.from_numpy(scores).to(torch.float32)
  new_fields['pred_classes'] = torch.from_numpy(labels).to(torch.int64)

  new_Instances = detectron2.structures.instances.Instances(image_size=isize, **new_fields)
  output = {}
  output['instances'] = new_Instances

  return [output] 

if __name__=='__main__':

	models = []
	models.append(torch.load("/content/datasets/coco/faster_rcnn_R_101_FPN_3x.pth"))
	models.append(torch.load("/content/datasets/coco/faster_rcnn_R_101_C4_3x.pth"))
	models.append(torch.load("/content/datasets/coco/faster_rcnn_R_101_DC5_3x.pth"))

	outputs = []
	for M in zip(*models):
	  outputs.append(ensemble(M))
