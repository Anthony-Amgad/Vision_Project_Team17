import numpy as np
import cv2

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

def rock_thresh(img, rgb_thresh=(85, 85, 30)):
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


# Apply the above functions in succession and update the Rover state accordingly
def perception_step(Rover):
    # Perform perception steps to update Rover()
    # TODO: 
    # NOTE: camera image is coming to you in Rover.img
    image = Rover.img


    # Define source and destination points for perspective transform (Warping)
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
    

    # Apply perspective transform

    warped = perspect_transform(image, source, destination)
    
    # Creating a mask so that nothing is detected out of our cone of vision

    mask = np.ones_like(image[:,:,0], np.uint8)
    mask = perspect_transform(mask, source, destination)

    # Apply color threshold to identify navigable terrain/obstacles/rock samples

    threshed = color_thresh(warped)
    obstic = cv2.bitwise_and(obst_thresh(warped), mask)
    rocks = rock_thresh(warped)

    # Update Rover.vision_image (the original image) for ease of user experience (this will be displayed on left side of screen)
    
    Rover.vision_image[:,:,2] = color_thresh(image)*255
    Rover.vision_image[:,:,0] = obst_thresh(image)*255
    Rover.vision_image[:,:,1] = rock_thresh(image)*255
    
    """ # Incase the wapred images are wanted for ease of debuggin
    Rover.vision_image[:,:,2] = threshed*255
    Rover.vision_image[:,:,0] = obstic*255
    Rover.vision_image[:,:,1] = rocks*255
    """

    # Convert map image pixel values to rover-centric coords
    
    xp, yp = rover_coords(threshed)
    oxp, oyp = rover_coords(obstic)
    rxp, ryp = rover_coords(rocks)

    # Applying a 70 pixel radius to cut off around the rover
    # To see a radius of 7 meters only 
    visdistance = np.sqrt(xp ** 2 + yp ** 2)
    xp = xp[visdistance<70]
    yp = yp[visdistance<70]

    visdistance = np.sqrt(oxp ** 2 + oyp ** 2)
    oxnp = oxp[visdistance<70]
    oynp = oyp[visdistance<70]

    # Convert rover-centric pixel values to world coordinates and update Rover worldmap (to be displayed on right side of screen)

    worldmap = Rover.worldmap
    # We need to make sure that the rover roll and pitch are around 0, and that the rover is not steering vigorously (for fidelity)
    # We need to make sure that the rover is not braking or there is any rock that is getting picked up (for fidelity)
    if ((Rover.pitch < 0.2 or Rover.pitch > 359.8) and (Rover.roll < 0.2 or Rover.roll > 359.8) and (abs(Rover.steer) <= 11) and (Rover.brake == 0) and (not Rover.picking_up)) or ((len(Rover.samples_angles) > 0) and (not Rover.picking_up) and (Rover.brake == 0)):
        
        obstacle_x_world, obstacle_y_world = pix_to_world(oxnp,oynp,Rover.pos[0],Rover.pos[1],Rover.yaw,worldmap.shape[0],2*dst_s)
        Rover.worldmap[obstacle_y_world, obstacle_x_world, 0] = 255
        Rover.worldmap[obstacle_y_world, obstacle_x_world, 2] = 0

        navigable_x_world, navigable_y_world = pix_to_world(xp,yp,Rover.pos[0],Rover.pos[1],Rover.yaw,worldmap.shape[0],2*dst_s)
        Rover.worldmap[navigable_y_world, navigable_x_world, 2] = 255
        Rover.worldmap[navigable_y_world, navigable_x_world, 0] = 0

        rock_x_world, rock_y_world = pix_to_world(rxp,ryp,Rover.pos[0],Rover.pos[1],Rover.yaw,worldmap.shape[0],2*dst_s)
        Rover.worldmap[rock_y_world, rock_x_world, 1] = 255

        Rover.worldmap = np.clip(Rover.worldmap, 0, 255)


    # Convert rover-centric pixel positions to polar coordinates and update Rover pixel distances and angles
      
    dist, angles = to_polar_coords(xp, yp)
    Rover.nav_dists = dist
    Rover.nav_angles = angles

    # First vicinity of rocks to enter 'found' mode
    rvisdistance = np.sqrt(rxp ** 2 + ryp ** 2)
    rxdp1 = rxp[rvisdistance<36]
    rydp1 = ryp[rvisdistance<36]

    # All rocks to move towards them when a rock is found
    rxp, ryp = rover_coords(rocks)

    # Second vicinity of rocks to enter 'found' mode
    rocks[:,:150] = 0
    rxp2, ryp2 = rover_coords(rocks)
    rvisdistance = np.sqrt(rxp2 ** 2 + ryp2 ** 2)
    rxdp2 = rxp2[rvisdistance<80]
    rydp2 = ryp2[rvisdistance<80]

    # Updating the Rover state
    rdist2, rangles2 = to_polar_coords(rxdp2,rydp2)
    rdist1, rangles1 = to_polar_coords(rxdp1,rydp1)
    rdist3, rangles3 = to_polar_coords(rxp, ryp)
    Rover.samples_angles = rangles1
    Rover.samples_dists = rdist1
    Rover.samples_angles2 = rangles2
    Rover.samples_dists2 = rdist2
    Rover.samples_dists3 = rdist3
    Rover.samples_angles3 = rangles3

    ## Obsticale vicinities
    visdistance = np.sqrt(oxp ** 2 + oyp ** 2)

    # This vicinity is to go to the average direction instead of hugging the wall
    oxdp = oxp[visdistance<37]
    oydp = oyp[visdistance<37]

    # This vicinity is to come to a complete stop
    oxdp2 = oxp[visdistance<13]
    oydp2 = oyp[visdistance<13]

     # Updating the Rover state
    odist, oangles = to_polar_coords(oxdp,oydp)
    odist2, oangles2 = to_polar_coords(oxdp2,oydp2)
    Rover.obst_angles = oangles
    Rover.obst_dists = odist
    Rover.obst_angles2 = oangles2
    Rover.obst_dists2 = odist2
    
    
    
    ######## DEBUGGING MODE
    
    #### SET TO TRUE IF YOU WANT DEBUGGING MODE ACTIVE
    dbugmode = False

    if dbugmode:
        arrow_length = 100
        mean_dir = np.mean(angles)
        stdev = 0.8 * np.std(angles * 180/np.pi) * (np.pi/180)
        if len(rangles3) > 0:
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
            pimg = np.zeros((321,161,3), np.uint8)
            oxpi = np.int_(oxnp)
            oypi = np.int_(oynp)
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


            if (Rover.mode != 'found') and (Rover.mode != 'pick') and (Rover.mode != 'lockedin'):
                if np.min(np.absolute(oangles * 180/np.pi)) > 16:
                    pimg = cv2.line(pimg, (0,160), (int(x_new_arrow), int(y_new_arrow)+160), color=(255,255,255), thickness=5)
                    pimg = cv2.line(pimg, (0,160), (int(x_arrow), int(y_arrow)+160), color=(255,255,0), thickness=5)
                else:
                    pimg = cv2.line(pimg, (0,160), (int(x_new_arrow), int(y_new_arrow)+160), color=(255,255,0), thickness=5)
                    pimg = cv2.line(pimg, (0,160), (int(x_arrow), int(y_arrow)+160), color=(255,255,255), thickness=5)
            else:
                if len(rangles3) > 0:
                    pimg = cv2.line(pimg, (0,160), (int(xr_arrow), int(yr_arrow)+160), color=(0,255,255), thickness=5) 
            cv2.imshow("Polar Image", pimg)
            
        except:
            print("no blues")
            if x_new_arrow == x_new_arrow and y_new_arrow == y_new_arrow and x_arrow == x_arrow and y_arrow == y_arrow:
                pimg = cv2.line(pimg, (0,160), (int(x_new_arrow), int(y_new_arrow)+160), color=(255,255,255), thickness=5)
                pimg = cv2.line(pimg, (0,160), (int(x_arrow), int(y_arrow)+160), color=(255,255,0), thickness=5)
            cv2.imshow("Polar Image", pimg)
        
        cv2.waitKey(1)
    
    return Rover