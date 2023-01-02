import cv2
import numpy as np
import csv
import matplotlib.image as mpimg

# Identify pixels above the threshold
# Threshold of RGB > 160 does a nice job of identifying ground pixels only
def color_thresh(img, rgb_thresh=(160, 160, 160)):
    # Create an array of zeros same xy size as img, but single channel
    color_select = np.zeros_like(img[:,:,0])
    # Require that each pixel be above all three threshold values in RGB
    # above_thresh will now contain a boolean array with "True"
    # where threshold was met
    above_thresh = (img[:,:,0] > rgb_thresh[0]) \
                & (img[:,:,1] > rgb_thresh[1]) \
                & (img[:,:,2] > rgb_thresh[2])
    # Index the array of zeros with the boolean array and set to 1
    color_select[above_thresh] = 1
    # Return the binary image
    return color_select

# Define a function to convert from image coords to rover coords
def rover_coords(binary_img):
    # Identify nonzero pixels
    ypos, xpos = binary_img.nonzero()
    # Calculate pixel positions with reference to the rover position being at the 
    # center bottom of the image.  
    x_pixel = -(ypos - binary_img.shape[0]).astype(np.float)
    y_pixel = -(xpos - binary_img.shape[1]/2 ).astype(np.float)
    return x_pixel, y_pixel


# Define a function to convert to radial coords in rover space
def to_polar_coords(x_pixel, y_pixel):
    # Convert (x_pixel, y_pixel) to (distance, angle) 
    # in polar coordinates in rover space
    # Calculate distance to each pixel
    dist = np.sqrt(x_pixel**2 + y_pixel**2)
    # Calculate angle away from vertical for each pixel
    angles = np.arctan2(y_pixel, x_pixel)
    return dist, angles

# Define a function to map rover space pixels to world space
def rotate_pix(xpix, ypix, yaw):
    # Convert yaw to radians
    yaw_rad = yaw * np.pi / 180
    xpix_rotated = (xpix * np.cos(yaw_rad)) - (ypix * np.sin(yaw_rad))
                            
    ypix_rotated = (xpix * np.sin(yaw_rad)) + (ypix * np.cos(yaw_rad))
    # Return the result  
    return xpix_rotated, ypix_rotated

def translate_pix(xpix_rot, ypix_rot, xpos, ypos, scale): 
    # Apply a scaling and a translation
    xpix_translated = (xpix_rot / scale) + xpos
    ypix_translated = (ypix_rot / scale) + ypos
    # Return the result  
    return xpix_translated, ypix_translated


# Define a function to apply rotation and translation (and clipping)
# Once you define the two functions above this function should work
def pix_to_world(xpix, ypix, xpos, ypos, yaw, world_size, scale):
    # Apply rotation
    xpix_rot, ypix_rot = rotate_pix(xpix, ypix, yaw)
    # Apply translation
    xpix_tran, ypix_tran = translate_pix(xpix_rot, ypix_rot, xpos, ypos, scale)
    # Perform rotation, translation and clipping all at once
    x_pix_world = np.clip(np.int_(xpix_tran), 0, world_size - 1)
    y_pix_world = np.clip(np.int_(ypix_tran), 0, world_size - 1)
    # Return the result
    return x_pix_world, y_pix_world

# Define a function to perform a perspective transform
def perspect_transform(img, src, dst):
           
    M = cv2.getPerspectiveTransform(src, dst)
    warped = cv2.warpPerspective(img, M, (img.shape[1], img.shape[0]))# keep same size as input image
    
    return warped

def obst_thresh(img, rgb_thresh=(100, 100, 100)):
    # Create an array of zeros same xy size as img, but single channel
    # obs will now contain a boolean array with "True"
    # where threshold was met
    color_select = np.zeros_like(img[:,:,0])
    obs = (img[:,:,0] < rgb_thresh[0]) \
                & (img[:,:,1] < rgb_thresh[1]) \
                & (img[:,:,2] < rgb_thresh[2])
    # Index the array of zeros with the boolean array and set to 1
    color_select[obs] = 1
    # Return the binary image
    return color_select

def rock_thresh(img, rgb_thresh=(95, 95, 20)):
    # Create an array of zeros same xy size as img, but single channel
    # rocks will now contain a boolean array with "True"
    # where threshold was met
    color_select = np.zeros_like(img[:,:,0])
    rocks = (img[:,:,0] > rgb_thresh[0]) \
                & (img[:,:,1] > rgb_thresh[1]) \
                & (img[:,:,2] < rgb_thresh[2])
    # Index the array of zeros with the boolean array and set to 1
    color_select[rocks] = 1
    # Return the binary image
    return color_select


csvpath = '../test_dataset/robot_log.csv'
ground_truth = mpimg.imread('../calibration_images/map_bw.png')
ground_truth = cv2.merge((ground_truth,ground_truth,ground_truth))*0

with open(csvpath, 'r') as file:
  csvreader = list(csv.reader(file, delimiter=';'))
  count = 0
  for row in csvreader[1:]:
    count += 1
    impath = row[0]
    imsteer = np.float64(row[1])
    imbrake = np.float64(row[3])
    imxp = np.float64(row[5])
    imyp = np.float64(row[6])
    impitch = np.float64(row[7])
    imyaw = np.float64(row[8])
    imroll = np.float64(row[9])
    image = mpimg.imread(impath)
    vision_image = cv2.imread(impath)
    
    dst_s = 5
    bottom_offset = 5
    
    source = np.float32([[14, 140],
                     [303, 140],
                     [200, 96],
                     [118, 96]])
    
    destination = np.float32([[image.shape[1] / 2 - dst_s, image.shape[0] - bottom_offset],
                          [image.shape[1] / 2 + dst_s, image.shape[0] - bottom_offset],
                          [image.shape[1] / 2 + dst_s, image.shape[0] - 2*dst_s - bottom_offset],
                          [image.shape[1] / 2 - dst_s, image.shape[0] - 2*dst_s - bottom_offset]])
    
    # 2) Apply perspective transform

    warped = perspect_transform(image, source, destination)
    
    mask = np.ones_like(image[:,:,0], np.uint8)
    mask = perspect_transform(mask, source, destination)

    # 3) Apply color threshold to identify navigable terrain/obstacles/rock samples

    threshed = color_thresh(warped)
    obstic = cv2.bitwise_and(obst_thresh(warped), mask)
    rocks = rock_thresh(warped)

    # 4) Update Rover.vision_image (this will be displayed on left side of screen)

    vision_image[:,:,0] = color_thresh(image)*255
    vision_image[:,:,2] = obst_thresh(image)*255
    vision_image[:,:,1] = rock_thresh(image)*255

    # 5) Convert map image pixel values to rover-centric coords

    xp, yp = rover_coords(threshed)
    oxp, oyp = rover_coords(obstic)
    rxp, ryp = rover_coords(rocks)

    visdistance = np.sqrt(xp ** 2 + yp ** 2)
    xp = xp[visdistance<70]
    yp = yp[visdistance<70]

    visdistance = np.sqrt(oxp ** 2 + oyp ** 2)
    oxnp = oxp[visdistance<70]
    oynp = oyp[visdistance<70]

    # 6) Convert rover-centric pixel values to world coordinates

    dist, angles = to_polar_coords(xp, yp)

    visdistance = np.sqrt(oxp ** 2 + oyp ** 2)
    oxdp = oxp[visdistance<31]
    oydp = oyp[visdistance<31]
    odist, oangles = to_polar_coords(oxdp,oydp)

    rvisdistance = np.sqrt(rxp ** 2 + ryp ** 2)
    rxdp1 = rxp[rvisdistance<36]
    rydp1 = ryp[rvisdistance<36]
    rdist1, rangles1 = to_polar_coords(rxdp1,rydp1)
    rdist3, rangles3 = to_polar_coords(rxp, ryp)

    # 7) Update Rover worldmap (to be displayed on right side of screen)

    if ((impitch < 0.2 or impitch > 359.8) and (imroll < 0.2 or imroll > 359.8) and (abs(imsteer) <= 11) and (imbrake == 0)) or ((len(rangles1) > 0) and (imbrake == 0)):
        
        obstacle_x_world, obstacle_y_world = pix_to_world(oxnp,oynp,imxp,imyp,imyaw,ground_truth.shape[0],2*dst_s)
        ground_truth[obstacle_y_world, obstacle_x_world, 2] = 255
        ground_truth[obstacle_y_world, obstacle_x_world, 0] = 0

        rock_x_world, rock_y_world = pix_to_world(rxp,ryp,imxp,imyp,imyaw,ground_truth.shape[0],2*dst_s)
        ground_truth[rock_y_world, rock_x_world, 1] = 255

        navigable_x_world, navigable_y_world = pix_to_world(xp,yp,imxp,imyp,imyaw,ground_truth.shape[0],2*dst_s)
        ground_truth[navigable_y_world, navigable_x_world, 0] = 255
        ground_truth[navigable_y_world, navigable_x_world, 2] = 0
    

    arrow_length = 100
    mean_dir = np.mean(angles)
    stdev = 0.8 * np.std(angles * 180/np.pi) * (np.pi/180)
    mean_rdir = np.mean(rangles3)
    xr_arrow = arrow_length * np.cos(mean_rdir)
    yr_arrow = arrow_length * np.sin(mean_rdir)
    x_arrow = arrow_length * np.cos(mean_dir)
    y_arrow = arrow_length * np.sin(mean_dir)
    x_new_arrow = arrow_length * np.cos(mean_dir - stdev)
    y_new_arrow = arrow_length * np.sin(mean_dir - stdev)
    try:
        cv2.imshow('Original Image', image)
        cv2.imshow('Warped Image', warped)
        cv2.imshow('Navigatabile Warped Terrain Image', threshed*255)
        cv2.imshow('Obstical Warpeed Terrain Image', obstic*255)
        cv2.imshow('Rock Warped Terrain Image', rocks*255)
        cv2.imwrite('RDAoutput/Warped Images/img'+str(count)+'.jpg', warped)
        cv2.imwrite('RDAoutput/Navigatabile Warped Terrain Images/img'+str(count)+'.jpg', threshed*255)
        cv2.imwrite('RDAoutput/Obstical Warpeed Terrain Images/img'+str(count)+'.jpg', obstic*255)
        cv2.imwrite('RDAoutput/Rock Warped Terrain Images/img'+str(count)+'.jpg', rocks*255)
        pimg = np.zeros((321,161,3), np.uint8)
        oxpi = np.int_(oxp)
        oypi = np.int_(oyp)
        for i in range(len(oxpi)):
            pimg = cv2.circle(pimg, (oxpi[i],oypi[i]+160), radius=0, color=(0,0,255), thickness=1)
        rxpi = np.int_(rxp)
        rypi = np.int_(ryp)
        for i in range(len(rxpi)):
            pimg = cv2.circle(pimg, (rxpi[i],rypi[i]+160), radius=0, color=(0,255,0), thickness=1)
        xpi = np.int_(xp)
        ypi = np.int_(yp)
        for i in range(len(xpi)):
            pimg = cv2.circle(pimg, (xpi[i],ypi[i]+160), radius=0, color=(255,0,0), thickness=1)
        if (len(rangles3) > 0):
            pimg = cv2.line(pimg, (0,160), (int(xr_arrow), int(yr_arrow)+160), color=(0,255,255), thickness=5)
        if np.min(np.absolute(oangles * 180/np.pi)) > 16:
            pimg = cv2.line(pimg, (0,160), (int(x_new_arrow), int(y_new_arrow)+160), color=(255,255,255), thickness=5)
            pimg = cv2.line(pimg, (0,160), (int(x_arrow), int(y_arrow)+160), color=(255,255,0), thickness=5)
        else:
            pimg = cv2.line(pimg, (0,160), (int(x_new_arrow), int(y_new_arrow)+160), color=(255,255,0), thickness=5)
            pimg = cv2.line(pimg, (0,160), (int(x_arrow), int(y_arrow)+160), color=(255,255,255), thickness=5)
        cv2.imshow("Polar Image", pimg)
        cv2.imwrite("RDAoutput/Polar Images/img"+str(count)+'.jpg', pimg)

        
    except:
        print("no blues")
        pimg = cv2.line(pimg, (0,160), (int(x_new_arrow), int(y_new_arrow)+160), color=(255,255,255), thickness=5)
        pimg = cv2.line(pimg, (0,160), (int(x_arrow), int(y_arrow)+160), color=(255,255,0), thickness=5)
        cv2.imshow("Polar Image", pimg)
        cv2.imwrite("RDAoutput/Polar Images/img"+str(count)+'.jpg', pimg)


    cv2.imshow('Map Image', ground_truth)
    cv2.imshow('Vision Image', vision_image)
    cv2.imwrite('RDAoutput/Map Images/img'+str(count)+'.jpg', ground_truth)
    cv2.imwrite('RDAoutput/Vision Images/img'+str(count)+'.jpg', vision_image)
    cv2.waitKey(10)

