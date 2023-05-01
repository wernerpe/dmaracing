# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.
import csv

class RaceTrack(object):
	def __init__(self, 
				 name, 
				 xy, 
				 bounds, 
				 full_zoom=0.12,
				 max_episode_steps=1000):
		self.name = name
		self.xy = xy
		self.bounds = bounds
		self.full_zoom = full_zoom
		self.max_episode_steps = max_episode_steps
	
	def save_coordinates_to_csv(self):
		filename = f"{self.name}_coordinates.csv"
		with open(filename, 'w', newline='') as file:
			writer = csv.writer(file)
			for coordinate in self.xy:
				writer.writerow([0.1*coordinate[0], 0.1*coordinate[1]])

