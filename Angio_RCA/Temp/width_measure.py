import cv2
from skimage.morphology import skeletonize, dilation,disk, thin
from skimage.filters import roberts
import matplotlib.pyplot as plt
import numpy as np
from scipy import spatial
from scipy.stats import linregress, zscore
from scipy.signal import savgol_filter, medfilt
from scipy.ndimage.morphology import binary_fill_holes
import os
import math
from bresenham import bresenham
import tifffile

def prune(inds):
	## PRUNING STEP. remove branches with fewer than 25 points
	n4neigh = [spatial.cKDTree(inds).query(pt, k=4,p=2, distance_upper_bound=1.5) for pt in inds]
	n4dist = [row[0] for row in n4neigh]
	n4pts = [row[1] for row in n4neigh]

	# find end points
	end_pts = []
	for i in range(len(n4dist)):		
		valids = n4dist[i][np.where(n4dist[i]<1.5)] 
		if len(valids) == 2: # an endpoint
			end = n4pts[i][np.where(n4dist[i]==0)]
			end_pts.append(end[0])

	# find all the coordinates from an end point to the 
	# nearest bifurcation point
	branches = {}
	pts_to_remove = np.empty(0)

	for pt in end_pts:
		branch = np.empty(0)
		prev_pt, curr_pt = pt, pt
		while True:
			valids = n4pts[curr_pt][np.where(n4dist[curr_pt]<1.5)[0]]
			if len(valids) > 3: # bifurcation point
				break
			if len(valids) == 2 and curr_pt != pt: # another end point
				break
			branch = np.append(branch,curr_pt)
			next_pt = valids[np.where((valids!=curr_pt) & (valids!=prev_pt))[0]][0]
			prev_pt = curr_pt
			curr_pt = next_pt
		if len(branch) > 25: # skip main branch
			continue
		pts_to_remove = np.append(pts_to_remove,branch)

	inds = np.delete(inds,pts_to_remove.astype(np.int), 0)

	## RE-SKELETONIZE. need to remove one-pixel branches
	pruned = np.zeros((512,512))
	for pt in range(inds.shape[0]):
		pruned[inds[pt,0],inds[pt,1]] = 255
	pruned = skeletonize(pruned>0,method='lee')

	inds = np.where(pruned>0)
	ind_row = np.expand_dims(inds[0],axis=1)
	ind_col = np.expand_dims(inds[1],axis=1)
	inds = np.concatenate((ind_row, ind_col), axis=1)

	return inds


class Width_Measure(object):
	# perform mask operations
	def __init__(self,file_path):
		mask = cv2.imread(file_path, 0)

		_, labeled, stats, _ = cv2.connectedComponentsWithStats(mask, connectivity=8)
		largest_label = np.where(stats[1:,4] == max(stats[1:,4]))[0] #zeroth label is background
		mask = labeled== (largest_label+1) #zeroth label here is actually first label
		mask = binary_fill_holes(mask)
		mask = mask.astype('uint8') * 255

		inds = np.where(mask>0)
		self.inds_mask = [tuple((inds[0][index],inds[1][index])) for index in range(inds[0].shape[0])]

		# find the centerline
		mask = cv2.dilate(mask,np.ones((3,3),np.uint8),iterations=1)
		mask = cv2.erode(mask,np.ones((3,3),np.uint8),iterations=1)
		skeleton = skeletonize(mask>0)

		self.img_dimensions = skeleton.shape

		inds = np.where(skeleton>0)
		ind_row = np.expand_dims(inds[0],axis=1)
		ind_col = np.expand_dims(inds[1],axis=1)
		inds = np.concatenate((ind_row, ind_col), axis=1)

		self.inds_first_skel = inds
		
		## PRUNING
		prune_again, loop = True, 0
		while prune_again:
			prune_again = False
			loop += 1
			inds = prune(inds)
			n4neigh = [spatial.cKDTree(inds).query(pt, k=4, p=2, distance_upper_bound=1.5) for pt in inds]
			n4dist = [row[0] for row in n4neigh]

			for i in range(len(n4dist)):
				valids = n4dist[i][np.where(n4dist[i]<1.5)]
				# check if there are any one-point branch that may mess up ordering
				if len(valids) == 4:
					prune_again = True

			if loop == 100:
				print("Warning: Remaining branches may mess up ordering.")
				print(file_path)
				break

		self.inds_pruned = inds

		## CHECKPOINT. find starting point of centerline
		n4neigh = [spatial.cKDTree(inds).query(pt, k=4,p=2, distance_upper_bound=1.5) for pt in inds]
		n4dist = [row[0] for row in n4neigh]

		start_pt_found_flag = False
		for i in range(len(n4dist)):
			valids = n4dist[i][np.where(n4dist[i]<1.5)]
			if len(valids) == 2 and start_pt_found_flag == False:
				start_pt = i
				start_pt_found_flag = True
			# check if there are any one-point branch that may mess up ordering
			if len(valids) == 4:
				print("Warning: Remaining branches may mess up ordering.")
				print(file_path)
				prune_again = True

		## ORDERING STEP. need to get correct sequence of points to generate deformation control points.
		nearest_neighbors = [spatial.cKDTree(inds).query(pt, k=3) for pt in inds]
		distances = [0]
		seq = []

		# find the next point on the centerline
		neighbors = list(nearest_neighbors[start_pt][1])
		neighbors.remove(start_pt)
		neighbors.remove(nearest_neighbors[start_pt][1][np.where(nearest_neighbors[start_pt][0]>1.5)[0][0]])

		prev_pt = start_pt
		curr_pt = neighbors[0]
		seq.append(start_pt)
		seq.append(curr_pt)

		for i in range(inds.shape[0]-2):
			neighbors = list(nearest_neighbors[curr_pt][1])
			# convert list to an int by indexing into [0].
			next_pt = [pt for pt in neighbors if pt not in (curr_pt,prev_pt)][0]
			distances.append(np.sqrt((inds[curr_pt,0]-inds[next_pt,0])**2 + (inds[curr_pt,1]-inds[next_pt,1])**2))
			seq.append(next_pt)
		
			prev_pt = curr_pt
			curr_pt = next_pt

		if len(seq) != inds.shape[0]:
			print("Warning: Flaw in ordering")
			print(file_path)

		self.inds_ordered = inds		# inds_ordered is not ordered!!!!

		## SMOOTH CENTERLINE. apply savgot_filter over sliding window
		# ctr_row = np.reshape(inds[seq,0],(len(seq),1))
		# ctr_col = np.reshape(inds[seq,1],(len(seq),1))
		# ctr_row[:,0] = savgol_filter(ctr_row[:,0],3,1,mode="constant")
		# ctr_col[:,0] = savgol_filter(ctr_col[:,0],3,1,mode="constant")

		# pruned = np.zeros(skeleton.shape)	
		# pruned[ctr_row, ctr_col]=255
		
		# inds = np.where(pruned>0)
		# ind_row = np.expand_dims(inds[0],axis=1)
		# ind_col = np.expand_dims(inds[1],axis=1)
		# inds = np.concatenate((ind_row, ind_col), axis=1)

		# self.inds_smooth = inds

		# mask = np.zeros(self.img_dimensions)
		# for pt in range(len(self.inds_mask)):
		# 	mask[self.inds_mask[pt][0],self.inds_mask[pt][1]] = 255
		# plt.imshow(mask,cmap='gray')

		# # draw skeleton
		# # borders = np.zeros((self.img_dimensions))
		# # for p_ind in range(len(self.inds_smooth)):
		# # 	borders[self.inds_smooth[p_ind][0],self.inds_smooth[p_ind][1]] = 255

		# # plt.imshow(borders,alpha=0.5)
		# plt.show()
		# if len(seq) != inds.shape[0]:
		# 	print("Warning: Savgol_filter error")
		# 	print(file_path)
		self.illustrated_normal = []
		self.illustrated_bounds = []

		# ## VESSEL WIDTH MEASUREMENT STEP. draw normal at one every few points at centerline,
		# # then on the normal, find the coordinates of the mask's border. vessel width is 
		# # calculated using euclidean distance.
		points = [tuple((inds[seq[index],0],inds[seq[index],1])) for index in range(8,len(seq)-8)] 

		inds_chosen_centerline = points[:]
		bnd, vessel_widths = [], []

		for p_ind, point in enumerate(points):
			# find the gradient of tangent
			if p_ind <= 2:
				tgnt = points[p_ind:(p_ind+7)] 
			elif p_ind >= len(points)-3: 
				tgnt = points[(p_ind-7):p_ind]
			else:
				tgnt = points[(p_ind-3):(p_ind+4)]
			slope, intercept, rvalue, pvalue, stderr = linregress(x=tgnt,y=None)

			# find the gradient of normal 
			horz_slope_flag = False
			if slope == 0: 
				if stderr == 0:
					horz_slope_flag = True # normal is a horizontal line
				else:
					normal_grad = 0 # normal is a vertical line
			elif math.isnan(slope): 
				normal_grad = 0 # normal is a vertical line
			else:
				normal_grad = -1 / slope	

			# bnd1 and bnd2 are coordinates of the bounds of the line profile being assessed
			if horz_slope_flag:
				bnd1 = tuple((point[0], point[1]-25))
				bnd2 = tuple((point[0], point[1]+25))
			else:
				limit = int(25/math.log(abs(normal_grad)+2,2)) # dist btw bounds at any point shd be around the same
				bnd1 = tuple((int(round(point[0]-limit)), int(round(point[1]+(normal_grad*(-limit))))))
				bnd2 = tuple((int(round(point[0]+limit)), int(round(point[1]+(normal_grad*limit)))))

			# find points on the mask's border that are within the bounds
			line = list(bresenham(bnd1[0],bnd1[1],bnd2[0],bnd2[1])) # listing the points btw the 2 bounds
			bnd1 = next(pt for pt in line if pt in self.inds_mask)
			bnd2 = next(pt for pt in line[::-1] if pt in self.inds_mask)
			self.illustrated_normal.append(line)
			self.illustrated_bounds.append(list([bnd1,bnd2]))

			bnd = bnd + line[:line.index(bnd1)+1] + line[line.index(bnd2):] 
			vessel_widths.append(math.sqrt((bnd2[1] - bnd1[1])**2 + (bnd2[0] - bnd1[0])**2)) 

		## REMOVE OUTLIERS. using z-score i.e. number of std from mean
		z = zscore(vessel_widths)
		vessel_widths = [vessel_widths[w] for w in np.where(z<3)[0]]

		self.inds_centerline = inds_chosen_centerline
		self.inds_normal = bnd
		self.vessel_widths = vessel_widths
		self.distances = distances
		self.file = file_path.split('/')[-1]

	def draw_first_skel(self,file_path=None):
		# draw mask
		mask = np.zeros(self.img_dimensions)
		for pt in range(len(self.inds_mask)):
			mask[self.inds_mask[pt][0],self.inds_mask[pt][1]] = 255
		plt.imshow(mask,cmap='gray')

		# draw skeleton
		borders = np.zeros((self.img_dimensions))
		for p_ind in range(len(self.inds_first_skel)):
			borders[self.inds_first_skel[p_ind][0],self.inds_first_skel[p_ind][1]] = 255

		plt.imshow(borders,alpha=0.5)
		if file_path != None:
			plt.savefig(file_path, dpi=1200, format='pdf', bbox_inches='tight')
		else:
			plt.show()
		plt.close()

	def draw_pruned(self,file_path=None):
		# draw mask
		mask = np.zeros(self.img_dimensions)
		for pt in range(len(self.inds_mask)):
			mask[self.inds_mask[pt][0],self.inds_mask[pt][1]] = 255
		plt.imshow(mask,cmap='gray')

		# draw skeleton
		borders = np.zeros((self.img_dimensions))
		for p_ind in range(len(self.inds_pruned)):
			borders[self.inds_pruned[p_ind][0],self.inds_pruned[p_ind][1]] = 255

		plt.imshow(borders,alpha=0.5)
		if file_path != None:
			plt.savefig(file_path, dpi=1200, format='pdf', bbox_inches='tight')
		else:
			plt.show()
		plt.close()

	def draw_ordered(self,file_path=None):
		# draw mask
		mask = np.zeros(self.img_dimensions)
		for pt in range(len(self.inds_mask)):
			mask[self.inds_mask[pt][0],self.inds_mask[pt][1]] = 255
		plt.imshow(mask,cmap='gray')

		# draw skeleton
		borders = np.zeros((self.img_dimensions))
		for p_ind in range(len(self.inds_ordered)):
			borders[self.inds_ordered[p_ind][0],self.inds_ordered[p_ind][1]] = 255

		plt.imshow(borders,alpha=0.5)
		if file_path != None:
			plt.savefig(file_path, dpi=1200, format='pdf', bbox_inches='tight')
		else:
			plt.show()
		plt.close()

	def draw_normal(self,file_path=None):
		# draw mask
		mask = np.zeros(self.img_dimensions)
		for pt in range(len(self.inds_mask)):
			mask[self.inds_mask[pt][0],self.inds_mask[pt][1]] = 255
		plt.imshow(mask,cmap='gray')

		# draw overlay of centerline & border points
		borders = np.zeros(self.img_dimensions)
		for p_ind in range(len(self.inds_normal)):
			if ((self.inds_normal[p_ind][0] in range(0,self.img_dimensions[0])) and 
				(self.inds_normal[p_ind][1] in range(0,self.img_dimensions[1]))):
				borders[self.inds_normal[p_ind][0],self.inds_normal[p_ind][1]] = 255
		for p_ind in range(len(self.inds_centerline)):
			borders[self.inds_centerline[p_ind][0],self.inds_centerline[p_ind][1]] = 255

		plt.imshow(borders,alpha=0.5)
		if file_path != None:
			plt.savefig(file_path, dpi=1200, format='pdf', bbox_inches='tight')
		else:
			plt.show()
		plt.close()

	def draw_illustrated_normal(self,file_path=None):
		# draw mask
		mask = np.zeros((512,512,3))
		for i in range(512):
			for j in range(512):
				mask[i,j] = np.array([33,0,41]) / 255
		print(mask)
		for pt in range(len(self.inds_mask)):
			mask[self.inds_mask[pt][0],self.inds_mask[pt][1]] = np.array([162,128,170]) / 255
		for p_ind in range(len(self.inds_centerline)):
			mask[self.inds_centerline[p_ind][0],self.inds_centerline[p_ind][1]] = np.array([254,243,146]) / 255
		for p_ind in range(len(self.inds_centerline)):
			if p_ind % 50 == 0:
				for pt in self.illustrated_normal[p_ind]:
					mask[pt[0],pt[1]] = np.array([220,20,60]) / 255
				for pt in self.illustrated_bounds[p_ind]:
					mask[pt[0],pt[1]] = np.array([233,150,122]) / 255
		plt.imshow(mask,cmap='jet')

		if file_path != None:
			plt.savefig(file_path, dpi=1200, format='pdf', bbox_inches='tight')
		else:
			plt.show()
		plt.close()

	def draw_illustrated_stenosis(self,folder_path,file_path=None):
		image = cv2.imread(folder_path+'/'+self.file)
		min_coor = self.inds_centerline[self.vessel_widths.index(min(self.vessel_widths))]

		start_point = tuple((min_coor[1]-45,min_coor[0]-30))
		end_point = tuple((min_coor[1]+45,min_coor[0]+30))
		color = (255,0,0)
		thickness = 3

		image = cv2.rectangle(image, start_point, end_point, color, thickness) 
		if file_path != None:
			tifffile.imsave(file_path,image)
		else:
			cv2.imshow('image',image)
			cv2.waitKey(0)

	def get_vessel_widths(self):
		return self.vessel_widths

	def get_distances(self):
		return self.distances

	def plot_vessel_widths(self,file_path=None):
		plt.plot(range(len(self.vessel_widths)), self.vessel_widths)
		plt.ylabel('Number of pixels', fontsize=25)
		plt.ylim(0,30)
		plt.xlabel('Index of point on centerline', fontsize=25)
		plt.tick_params(axis='both', which='major', labelsize=15)
		#plt.rc('axes', labelsize=20) 
		if file_path != None:
			plt.savefig(file_path)
		else:
			plt.show()
		plt.close()




