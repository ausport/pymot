import json
import os
import csv
import numpy as np

"""
Import ground truth JSON from Microworks dataset, and convert to MOTA-friendly JSON groundtruth format.
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

def import_csv():
	_instances = []
	_frames = []
	_global_index = 0
	_min_area = 20
	_truncated_n = np.linspace(1, 750, num=200, dtype=int)  # How many samples from a truncated set?
	_set = "0_hockey_5mins.csv"

	# If running locally this should work.
	path = "{0}/Dropbox/_Microwork/Annotation_5min/{1}".format(os.path.expanduser("~"), _set)

	assert os.path.exists(path)

	image_width = 1920
	image_height = 1080

	json_frames = {"frames": [], "class": "video", "filename": path}

	with open(path, 'rt') as f:
		reader = csv.reader(f, delimiter=',', quotechar='"')
		sortedlist = sorted(reader, key=lambda _row: _row[4], reverse=False)
		_current_frame = 1
		_frame_labels = []



		_last_valid_frame_number = 0
		_is_new_frame_number = True
		frame = {}

		for row in sortedlist:

			if not row[0] == '#':

				print(row)
				_t = int(row[4].rsplit('.', 1)[0])

				if _t != _last_valid_frame_number:
					# New frame
					_last_valid_frame_number = _t

					# Create frame stub
					frame = {'timestamp': float(_t / 25.),
							'num': _t,
							"class": "frame",
							"annotations": []
							}

				# print(row)
				_top = int(row[9])
				_left = int(row[10])
				_width = int(row[11])
				_height = int(row[12])
				_bbox = [_left, _top, _width, _height]
				print(_bbox)
				new_annotation = {
					"dco": False,		# Not required for hypotheses..
					"height": _height,
					"width": _width,
					"id": row[2],
					"y": _top - _height / 2,
					"x": _left - _width / 2
				}

				print(new_annotation)

				frame["annotations"].append(new_annotation)




if __name__ == "__main__":

    # Initialise bottle server
    print("\n\n* * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * *")
    print("*\n* COMPILE MOTA GROUND TRUTH FILES!\n*")
    print("* * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * *")

    import_csv()


    print("All done...")
