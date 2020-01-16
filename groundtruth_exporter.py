import json
import os
import sys
import csv
import numpy as np
import cv2
from VideoTools import CV2Video
from PIL import ImageDraw, ImageFont
import motmetrics as mm


"""
Import ground truth JSON from Microworks dataset, and convert to MOTA-friendly JSON ground truth format.
"""

"""
	[
		{
			"frames": [
				{
					"timestamp": 0.054,
					"num": 0,
					"class": "frame",
					"annotations": [
						{
							"dco": true,
							"height": 31.0,
							"width": 31.0,
							"id": "sheldon",
							"y": 105.0,
							"x": 608.0
						}
					]
				},
				{
					"timestamp": 3.854,
					"num": 95,
					"class": "frame",
					"annotations": [
						{
							"dco": true,
							"height": 31.0,
							"width": 31.0,
							"id": "sheldon",
							"y": 105.0,
							"x": 608.0
						},
						{
							"dco": true,
							"height": 38.0,
							"width": 29.0,
							"id": "leonard",
							"y": 145.0,
							"x": 622.0
						}
					]
				}
			],
			"class": "video",
			"filename": "/cvhci/data/multimedia/bigbangtheory/bbt_s01e01/bbt_s01e01.idx"
		}
	]

"""


lut = [(230, 25, 75), (60, 180, 75), (255, 225, 25), (0, 130, 200), (245, 130, 48), (145, 30, 180), (70, 240, 240), (
				240, 50, 230), (210, 245, 60), (250, 190, 190), (0, 128, 128), (230, 190, 255), (170, 110, 40), (
				255, 250, 200), (128, 0, 0), (170, 255, 195), (128, 128, 0), (255, 215, 180), (0, 0, 128),
			(128, 128, 128), (
				255, 255, 255), (0, 0, 0), (255, 255, 75), (160, 180, 175), (55, 225, 125), (110, 130, 20),
			(45, 10, 148), (45, 0, 0), (0, 40, 240), (
				40, 50, 30), (10, 45, 60), (50, 90, 190), (10, 12, 12), (30, 90, 55), (70, 10, 40), (
				55, 50, 200), (28, 10, 10), (70, 255, 95), (0, 128, 0), (55, 15, 180), (10, 90, 128), (128, 0, 128),
			(
				255, 0, 255)]


def color_lut(k):
	while k > 40:
		k = - 40

	if k == -1:
		return [(0, 0, 0)]

	return lut[k]


def import_csv(annotate_video=False):
	_instances = []
	_frames = []
	_global_index = 0
	_min_area = 20
	_truncated_n = np.linspace(1, 750, num=200, dtype=int)  # How many samples from a truncated set?
	_set = "0_hockey_5mins.csv", "0_hockey_5mins.mp4", "0_hockey_5mins_annotated.mp4"

	# If running locally this should work.
	path_detections = "{0}/Dropbox/_Microwork/Annotation_5min/{1}".format(os.path.expanduser("~"), _set[0])
	path_video = "{0}/Dropbox/_Microwork/5min_tracking/{1}".format(os.path.expanduser("~"), _set[1])
	movie_out = "{0}/Dropbox/_Microwork/5min_tracking/{1}".format(os.path.expanduser("~"), _set[2])
	_video_writer = None
	mpv = None

	assert os.path.exists(path_detections), "Couldn't find detections."

	if annotate_video:
		assert os.path.exists(path_video), "Couldn't find video."

		mpv = CV2Video.CV2VideoObject(path_video, verbose_mode=False)
		mpv.showDetails()
		w = mpv.video_width()
		h = mpv.video_height()

		if mpv is None:
			print("Couldn't load the video...")
			sys.exit(1)

		# Create output video context
		assert len(movie_out) > 0

		print("Preparing video output to", movie_out)
		print("Input/Output Dimensions: {0} x {1}".format(w, h))
		print("**** THIS COULD TAKE A WHILE ***")
		fourcc = cv2.VideoWriter_fourcc('m', 'p', '4', 'v')
		_video_writer = cv2.VideoWriter(movie_out, fourcc, 25, (w, h), True)

	json_frames = {"frames": [], "class": "video", "filename": path_detections}

	with open(path_detections, 'rt') as f:
		reader = csv.reader(f, delimiter=',', quotechar='"')
		sortedlist = sorted(reader, key=lambda _row: _row[4], reverse=False)
		_current_frame = 1
		_frame_labels = []
		_last_valid_frame_number = 0
		_is_new_frame_number = True

		ImageFont.load_default()
		fnt = ImageFont.load_default()

		frame = {'timestamp': 0, 'num': 0, "class": "frame", "annotations": []}

		if annotate_video:
			# Pull first video frame
			pil_original_image = mpv.getFrame(0)
			draw = ImageDraw.Draw(pil_original_image)

		for row in sortedlist:

			if not row[0] == '#':

				_t = int(row[4].rsplit('.', 1)[0])
				if _t > _last_valid_frame_number:
					# New frame
					_last_valid_frame_number = _t
					_is_new_frame_number = True

					if len(frame["annotations"]) > 0:
						json_frames["frames"].append(frame)

						if annotate_video:
							_video_writer.write(cv2.cvtColor(np.array(pil_original_image), cv2.COLOR_RGB2BGR))

					# Create frame stub
					frame = {'timestamp': float(_t / 25.), 'num': _t, "class": "frame", "annotations": []}

					if annotate_video:
						# Pull video frame
						pil_original_image = mpv.getFrame(_t)
						draw = ImageDraw.Draw(pil_original_image)

				if _t > 25*60:
					break

				_top = int(row[9])
				_left = int(row[10])
				_width = int(row[11])
				_height = int(row[12])
				_bbox = [_left, _top, _width, _height]

				new_annotation = {
					"dco": False,		# Not required for hypotheses..
					"height": _height,
					"width": _width,
					"id": row[2],
					"y": _top - _height / 2,
					"x": _left - _width / 2
				}

				if annotate_video:
					_id = [int(s) for s in row[2].split() if s.isdigit()]
					if len(_id) > 0:
						_lut = color_lut(_id[0])
					else:
						_lut = ([0, 0, 0])

					draw.rectangle((_left, _top, _left + _width, _top + _height), outline=_lut)
					draw.text((_left, _top), row[2], font=fnt, fill=_lut)

				frame["annotations"].append(new_annotation)

		_path = "./Hockey_GroundTruth2.json"
		if _path != "":
			with open("{0}".format(_path), 'w') as _f:
				json.dump(json_frames, _f, indent=4)
				_f.write(os.linesep)

	return json_frames


if __name__ == "__main__":

	print("\n\n* * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * *")
	print("*\n* COMPILE MOTA GROUND TRUTH FILES!\n*")
	print("* * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * *")

	detections = import_csv(annotate_video=False)

	with open('groundtruth.json', 'r') as f:
		gt_dict = json.load(f)

	with open('hypotheses.json', 'r') as f:
		h_dict = json.load(f)

	id_labels = []

	# Create an accumulator that will be updated during each frame
	acc = mm.MOTAccumulator(auto_id=True)

	# Iterate through ground truth frames.
	for gt in gt_dict[0]["frames"]:

		_frame_num = gt['num']
		# print("*", _frame_num)
		id_labels = []

		a = np.empty((0, 4), int)
		# Append detections for this frame
		for annotation in gt['annotations']:
			_x = annotation['x']
			_y = annotation['y']
			_w = annotation['width']
			_h = annotation['height']
			print("Adding GT:", _x, _y, _w, _h)
			id_labels.append(annotation['id'])
			a = np.append(a, np.array([[_x, _y, _w, _h]]), axis=0)

		# Find detections matching frame id.
		hypothesis_frame = [x for x in h_dict[0]["frames"] if x['num'] == _frame_num][0]
		# print("Sanity", hypothesis_frame)
		b = np.empty((0, 4), int)
		h_id = []
		for hypotheses in hypothesis_frame['hypotheses']:
			_x = hypotheses['x']
			_y = hypotheses['y']
			_w = hypotheses['width']
			_h = hypotheses['height']
			print("Adding H:", _x, _y, _w, _h)
			b = np.append(b, np.array([[_x, _y, _w, _h]]), axis=0)
			h_id.append(hypotheses['id'])

		# print("H Frame {0}:\n {1}".format(_frame_num, hypothesis_frame))

		print("GT:")
		print(a)

		print("h_id")
		print(h_id)

		print("distances.iou_matrix")
		d = mm.distances.iou_matrix(a, b, max_iou=0.5)

		print(d)


		print(id_labels)

		# Call update once for per frame. For now, assume distances between
		# frame objects / hypotheses are given.
		acc.update(
			id_labels,                 # Ground truth objects in this frame
			h_id,                  # Detector hypotheses in this frame
			d
		)

		print(acc.events)  # a pandas DataFrame containing all events

		mh = mm.metrics.create()
		summary = mh.compute(acc, metrics=['num_frames', 'mota', 'motp'], name='acc')
		print(summary)

		# Generate summary statistics
		summary = mh.compute_many(
			[acc, acc.events.loc[0:1]],
			metrics=mm.metrics.motchallenge_metrics,
			names=['full', 'part'])

		# Make it look nice...
		strsummary = mm.io.render_summary(
			summary,
			formatters=mh.formatters,
			namemap=mm.io.motchallenge_metric_names
		)
		print(strsummary)