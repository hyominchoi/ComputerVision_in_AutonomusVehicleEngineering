import pickle
import cv2
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.image as mpimg
import glob
from lesson_functions import *
from scipy.ndimage.measurements import label


class Detector():

    def __init__(self):

        self.isCarDetected = False
        # number of frame lane has processed mod 10 (=n)
        self.num_frame = 0

        # input stream video image shape
        self.img_shape = (720, 1280)

        # load trained model
        self.svc = None# pickle.load(open('svc.p', 'rb'))
        self.X_scaler = None #pickle.load(open('X_scaler.p', 'rb'))

        # sliding window parameters
        self.x_start_stop = [[200, self.img_shape[1]],
                            [200, self.img_shape[1]],
                            [200, self.img_shape[1]],
                            [400, self.img_shape[1]]]
        self.y_start_stop = [[400, 656],
                            [400, 600],
                            [450, 550],
                            [400, 500],]
    
        self.scales = [1.5, 1.1, 0.9, 0.8]
        self.cells_per_step = 3

        # image features  parameters
        self.spatial_size = (32, 32)
        self.hist_bins = 32
        self.conv_color = 'RGB2HLS'
        
        # HOG reated parameters
        self.orient = 9
        self.pix_per_cell = 8
        self.cell_per_block = 2
        
        # Post Processing parameters
        self.threshold = 10 # heatmap threshold
        self.num_cars = 0
        self.frames_to_collect = 5
        self.heatmap = np.zeros(self.img_shape).astype(np.float)
        self.heatmap_recent = np.zeros(
                            (self.img_shape[0], 
                            self.img_shape[1], 
                            self.frames_to_collect)).astype(np.float)
        self.heatmap_mean = np.zeros(self.img_shape).astype(np.float)
        
    def reset(self):
        self.isCarDetected = False
        self.num_frame = 0
        self.num_cars = 0
        self.heatmap = np.zeros(self.img_shape).astype(np.float)
        self.heatmap_recent = np.zeros(
                            (self.img_shape[0], 
                            self.img_shape[1], 
                            self.frames_to_collect)).astype(np.float)
        self.heatmap_mean = np.zeros(self.img_shape).astype(np.float)

    def multiple_window_sizes_search(self, img):

        y_start_stop = self.y_start_stop
        x_start_stop = self.x_start_stop
        scales = self.scales
        self.num_frame = int(self.num_frame + 1)

        search_img =[]
        bboxes = []
        for i in range(len(scales)):
            ystart = y_start_stop[i][0]
            ystop = y_start_stop[i][1]
            xstart = x_start_stop[i][0]
            xstop = x_start_stop[i][1]
            scale = scales[i]
            car_boxes, draw_img = self.find_cars(img, 
                                    ystart, ystop,
                                    xstart, xstop,
                                    scale)
            search_img.append(draw_img)
            bboxes.extend(car_boxes)
        return bboxes, search_img


    def convert_color(self, img, conv):
        if conv == 'RGB2YCrCb':
            return cv2.cvtColor(img, cv2.COLOR_RGB2YCrCb)
        if conv == 'BGR2YCrCb':
            return cv2.cvtColor(img, cv2.COLOR_BGR2YCrCb)
        if conv == 'RGB2LUV':
            return cv2.cvtColor(img, cv2.COLOR_RGB2LUV)
        if conv == 'RGB2HLS':
            return cv2.cvtColor(img, cv2.COLOR_RGB2HLS)
        if conv == 'RGB2HSV':
            return cv2.cvtColor(img, cv2.COLOR_RGB2HSV)    
        
    def find_cars(self, img, ystart, ystop, xstart, xstop, scale):
        pix_per_cell = self.pix_per_cell
        cell_per_block = self.cell_per_block
        cells_per_step = self.cells_per_step
        orient = self.orient
        svc = self.svc
        X_scaler = self.X_scaler
        spatial_size = self.spatial_size
        hist_bins = self.hist_bins

        draw_img = np.copy(img)
        img = img.astype(np.float32)/255
        car_boxes = []
        
        img_tosearch = img[ystart:ystop, xstart:xstop,:]
        
        ctrans_tosearch = self.convert_color(img_tosearch, conv=self.conv_color)
        if scale != 1:
            imshape = ctrans_tosearch.shape
            ctrans_tosearch = cv2.resize(ctrans_tosearch, 
                                         (np.int(imshape[1]/scale), 
                                          np.int(imshape[0]/scale)))
            
        ch1 = ctrans_tosearch[:,:,0]
        ch2 = ctrans_tosearch[:,:,1]
        ch3 = ctrans_tosearch[:,:,2]

        # Define blocks and steps as above
        nxblocks = (ch1.shape[1] // pix_per_cell) - cell_per_block + 1
        nyblocks = (ch1.shape[0] // pix_per_cell) - cell_per_block + 1 
        nfeat_per_block = orient*cell_per_block**2
        
        # 64 was the orginal sampling rate, with 8 cells and 8 pix per cell
        window = 64
        nblocks_per_window = (window // pix_per_cell) - cell_per_block + 1
        # Instead of overlap ratio, define how many cells to step, use cells_per_step
        nxsteps = (nxblocks - nblocks_per_window) // cells_per_step
        nysteps = (nyblocks - nblocks_per_window) // cells_per_step
        
        # Compute individual channel HOG features for the entire image
        hog1 = get_hog_features(ch1, orient, pix_per_cell, cell_per_block, feature_vec=False)
        hog2 = get_hog_features(ch2, orient, pix_per_cell, cell_per_block, feature_vec=False)
        hog3 = get_hog_features(ch3, orient, pix_per_cell, cell_per_block, feature_vec=False)
        
        for xb in range(nxsteps):
            for yb in range(nysteps):
                ypos = yb*cells_per_step
                xpos = xb*cells_per_step
                # Extract HOG for this patch
                hog_feat1 = hog1[ypos:ypos+nblocks_per_window, xpos:xpos+nblocks_per_window].ravel() 
                hog_feat2 = hog2[ypos:ypos+nblocks_per_window, xpos:xpos+nblocks_per_window].ravel() 
                hog_feat3 = hog3[ypos:ypos+nblocks_per_window, xpos:xpos+nblocks_per_window].ravel() 
                hog_features = np.hstack((hog_feat1, hog_feat2, hog_feat3))

                xleft = xpos*pix_per_cell
                ytop = ypos*pix_per_cell

                # Extract the image patch
                subimg = cv2.resize(ctrans_tosearch[ytop:ytop+window, xleft:xleft+window], (64,64))
              
                # Get color features
                # spatial_features = bin_spatial(subimg, size=spatial_size)
                hist_features = color_hist(subimg, nbins=hist_bins)

                # Scale features and make a prediction
                #test_features = X_scaler.transform(np.hstack((spatial_features, hist_features, hog_features)).reshape(1, -1))    
                test_features = X_scaler.transform(np.hstack((hist_features, hog_features)).reshape(1, -1))    
                test_prediction = svc.predict(test_features)
                
                if test_prediction == 1:
                    xbox_left = np.int(xleft*scale)
                    ytop_draw = np.int(ytop*scale)
                    win_draw = np.int(window*scale)
                    cv2.rectangle(draw_img,
                                  (xbox_left+xstart, ytop_draw+ystart),
                                  (xbox_left+win_draw+xstart,ytop_draw+win_draw+ystart),
                                  (0,0,255),6) 
                    car_boxes.append(((xbox_left+xstart, ytop_draw+ystart), (xbox_left+win_draw+xstart, ytop_draw+win_draw+ystart)))
        
        if len(car_boxes) > 0 : 
            self.isCarDetected = True

        return car_boxes , draw_img


    def new_frame(self):
    	self.heatmap = np.zeros(self.img_shape).astype(np.float)

    def add_heat(self, bbox_list):
        for box in bbox_list:
            # Add += 1 for all pixels inside each bbox
            # Assuming each "box" takes the form ((x1, y1), (x2, y2))
            self.heatmap[box[0][1]:box[1][1], box[0][0]:box[1][0]] += 1
    
    def filter_false_positive(self):
        i = int(self.num_frame - 1) % self.frames_to_collect
        self.heatmap_recent[:,:,i] = self.heatmap

        if self.num_frame <= self.frames_to_collect:
            self.heatmap_mean = (self.heatmap) * self.frames_to_collect

        else: 
            product_heatmap = np.ones(self.img_shape).astype(np.float)
            sum_heatmap = np.zeros(self.img_shape).astype(np.float)
            for j in range(self.frames_to_collect):
                product_heatmap = product_heatmap * self.heatmap_recent[:,:,j]
                sum_heatmap = sum_heatmap + self.heatmap_recent[:,:,j]
        
            self.heatmap_mean = (sum_heatmap)


    def apply_threshold(self):
        # Zero out pixels below the threshold
        # self.heatmap_mean = self.heatmap
        self.heatmap_mean[self.heatmap_mean <= self.threshold] = 0
        self.heatmap_mean = np.clip(self.heatmap_mean, 0, 255)

    def apply_label(self):
        self.labels, self.num_cars = label(self.heatmap_mean)

    def draw_labeled_bboxes(self, img):
        labels = self.labels
        # Iterate through all detected cars
        for car_number in range(1, self.num_cars + 1):
            # Find pixels with each car_number label value
            nonzero = (labels == car_number).nonzero()
            # Identify x and y values of those pixels
            nonzeroy = np.array(nonzero[0])
            nonzerox = np.array(nonzero[1])
            # Define a bounding box based on min/max x and y
            bbox = ((np.min(nonzerox), np.min(nonzeroy)), 
                    (np.max(nonzerox), np.max(nonzeroy)))
            # Draw the box on the image
            cv2.rectangle(img, bbox[0], bbox[1], (0,0,255), 6)
        # Return the image
        return img



