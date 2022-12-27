import numpy as np
import time

# This is where you can build a decision tree for determining throttle, brake and steer 
# commands based on the output of the perception_step() function
def decision_step(Rover):

    if Rover.vel <= 0.1 and (Rover.total_time - Rover.stuck_time > 4):
        # Set mode to "stuck" and hit the brakes
        Rover.mode = 'stuck'
        Rover.throttle = 0
        # Set brake to stored brake value
        Rover.brake = Rover.brake_set
        Rover.steer = 0
        Rover.stuck_time = Rover.total_time
    #elif Rover.vel <= 0.1 and Rover.mode != 'instuck':
        #Rover.stuck_time = Rover.total_time
        #Rover.stuckmode = 'instuck'
    #elif Rover.vel > 0.1 and Rover.stuckmode == 'instuck':
        #Rover.stuckmode = 'notstuck'
    
    if (len(Rover.samples_angles) > 15) and Rover.mode != 'stuck' and Rover.mode != 'found1' and Rover.mode != 'found2' and Rover.mode != 'lockedin1' and Rover.mode != 'lockedin2' and Rover.mode != 'pick1' and Rover.mode != 'pick2':
        Rover.mode = 'found1'
        print('FOUND')
        Rover.stuck_time = Rover.total_time

    elif (len(Rover.samples_angles2) > 15) and Rover.mode != 'stuck' and Rover.mode != 'found2' and Rover.mode != 'lockedin1' and Rover.mode != 'lockedin2' and Rover.mode != 'pick1' and Rover.mode != 'pick2':
        Rover.mode = 'found2'
        print('FOUND')
        Rover.stuck_time = Rover.total_time

        """
        if np.mean(Rover.samples_dists) >= 10:  
            # If mode is forward, navigable terrain looks good 
            # and velocity is below max, then throttle 
            if Rover.vel < 0.5:
                # Set throttle value to throttle setting
                Rover.throttle = Rover.throttle_set*1.5
                Rover.brake = 0
                # Set steering to average angle clipped to the range +/- 15
                Rover.steer = np.clip(np.mean(Rover.samples_angles * 180/np.pi), -15, 15)
            else: # Else coast
                Rover.throttle = 0
                Rover.brake = 5
                Rover.steer = np.clip(np.mean(Rover.samples_angles * 180/np.pi), -15, 15)
                Rover.stuck_time = Rover.total_time
        elif np.mean(Rover.samples_dists) < 10:
                # Set mode to "stop" and hit the brakes!
                Rover.throttle = 0
                # Set brake to stored brake value
                Rover.brake = Rover.brake_set
                Rover.steer = 0
                Rover.mode = 'stop'
                if not Rover.picking_up:
                    Rover.send_pickup = True
        
    
    elif (len(Rover.samples_angles2) > 0) and Rover.mode != 'stuck':
        if np.mean(Rover.samples_dists2) >= 10:  
            # If mode is forward, navigable terrain looks good 
            # and velocity is below max, then throttle 
            if Rover.vel < 0.6:
                # Set throttle value to throttle setting
                Rover.throttle = Rover.throttle_set*1.5
                Rover.brake = 0
                # Set steering to average angle clipped to the range +/- 15
                Rover.steer = np.clip(np.mean(Rover.samples_angles2 * 180/np.pi), -15, 15)
            else: # Else coast
                Rover.throttle = 0
                Rover.brake = 1
                Rover.steer = np.clip(np.mean(Rover.samples_angles2 * 180/np.pi), -15, 15)
                Rover.stuck_time = Rover.total_time
            
        # If there's a lack of navigable terrain pixels then go to 'stop' mode
        elif np.mean(Rover.samples_dists2) < 10:
                # Set mode to "stop" and hit the brakes!
                Rover.throttle = 0
                # Set brake to stored brake value
                Rover.brake = Rover.brake_set
                Rover.steer = 0
                Rover.mode = 'stop'
                if not Rover.picking_up:
                    Rover.send_pickup = True
    """
    elif Rover.mode == 'found1':
        Rover.brake = 5
        Rover.stuck_time = Rover.total_time
        if Rover.vel == 0 and (Rover.pitch < 0.2 or Rover.pitch > 359.8) and (Rover.roll < 0.2 or Rover.roll > 359.8):
            Rover.mode = 'lockedin1'
    
    elif Rover.mode == 'found2':
        Rover.brake = 5
        Rover.stuck_time = Rover.total_time
        if Rover.vel == 0 and (Rover.pitch < 0.2 or Rover.pitch > 359.8) and (Rover.roll < 0.2 or Rover.roll > 359.8):
            Rover.mode = 'lockedin2'

    elif Rover.mode == 'lockedin1':
        mangle = np.mean(Rover.samples_angles * 180/np.pi)

        if (mangle < 1) and (mangle > -1):
            Rover.mode = 'pick1'
        else:
            Rover.brake = 0
            Rover.steer = np.clip(mangle, -15, 15)
            Rover.stuck_time = Rover.total_time

    elif Rover.mode == 'lockedin2':
        mangle = np.mean(Rover.samples_angles2 * 180/np.pi)

        if (mangle < 1) and (mangle > -1):
            Rover.mode = 'pick2'
        else:
            Rover.brake = 0
            Rover.steer = np.clip(mangle, -15, 15)
            Rover.stuck_time = Rover.total_time
    
    elif Rover.mode == 'pick1':     
        mangle = np.mean(Rover.samples_angles * 180/np.pi)
        if Rover.vel < Rover.max_vel:
            # Set throttle value to throttle setting
            Rover.throttle = Rover.throttle_set
            Rover.brake = 0
            # Set steering to average angle clipped to the range +/- 15
            #mangle = (np.mean(Rover.samples_angles2 * 180/np.pi)+np.mean(Rover.samples_angles * 180/np.pi))/2
            Rover.steer = np.clip(mangle, -15, 15)
        else: # Else coast
            Rover.throttle = 0
        
        if np.mean(Rover.samples_dists) < 12:
            # Set mode to "stop" and hit the brakes!
            Rover.throttle = 0
            # Set brake to stored brake value
            Rover.brake = Rover.brake_set
            Rover.steer = 0
            Rover.mode = 'stop'
            if not Rover.picking_up:
                Rover.send_pickup = True
        
    elif Rover.mode == 'pick2':
        mangle = np.mean(Rover.samples_angles2 * 180/np.pi)
        if Rover.vel < Rover.max_vel:
            # Set throttle value to throttle setting
            Rover.throttle = Rover.throttle_set
            Rover.brake = 0
            # Set steering to average angle clipped to the range +/- 15
            #mangle = (np.mean(Rover.samples_angles2 * 180/np.pi)+np.mean(Rover.samples_angles * 180/np.pi))/2
            Rover.steer = np.clip(mangle, -15, 15)
        else: # Else coast
            Rover.throttle = 0
        
        if np.mean(Rover.samples_dists2) < 12:
            # Set mode to "stop" and hit the brakes!
            Rover.throttle = 0
            # Set brake to stored brake value
            Rover.brake = Rover.brake_set
            Rover.steer = 0
            Rover.mode = 'stop'
            if not Rover.picking_up:
                Rover.send_pickup = True
        

    elif ((len(Rover.samples_angles2) > 0) or (len(Rover.samples_angles) > 0)) and Rover.mode == 'stuck':
        # if 1 sec passed go back to previous mode
        if Rover.total_time - Rover.stuck_time > 0.3:
            # Set throttle back to stored value
            Rover.throttle = Rover.throttle_set
            # Release the brake
            Rover.brake = 0
            Rover.mode = 'forward' 
        # Now we're stopped and we have vision data to see if there's a path forward
        else:
            Rover.throttle = 0
            # Release the brake to allow turning
            Rover.brake = 0
            Rover.steer = 15
    
    # Implement conditionals to decide what to do given perception data
    # Here you're all set up with some basic functionality but you'll need to
    # improve on this decision tree to do a good job of navigating autonomously!

    # Example:
    # Check if we have vision data to make decisions with
    elif Rover.nav_angles is not None:
        # Check for Rover.mode status
        if Rover.mode == 'forward': 
            Rover.send_pickup = False
            # Check the extent of navigable terrain
            if len(Rover.nav_angles) >= Rover.stop_forward:  
                # If mode is forward, navigable terrain looks good 
                # and velocity is below max, then throttle 
                if Rover.vel < Rover.max_vel:
                    # Set throttle value to throttle setting
                    Rover.throttle = Rover.throttle_set
                else: # Else coast
                    Rover.throttle = 0
                Rover.brake = 0
                # Set steering to average angle clipped to the range +/- 15
                try:
                    if (np.min(np.absolute(Rover.obst_angles * 180/np.pi)) > 16):
                        Rover.steer = np.clip(np.mean(Rover.nav_angles * 180/np.pi)-(0.8 * np.std(Rover.nav_angles * 180/np.pi)), -15, 15)
                        
                    else:
                        Rover.steer = np.clip(np.mean(Rover.nav_angles * 180/np.pi), -15, 15)
                        
                except:
                    print("Steering can't find near obst")
                    Rover.steer = np.clip(np.mean(Rover.nav_angles * 180/np.pi)-(0.8 * np.std(Rover.nav_angles * 180/np.pi)), -15, 15)
            # If there's a lack of navigable terrain pixels then go to 'stop' mode
            elif len(Rover.nav_angles) < Rover.stop_forward:
                    # Set mode to "stop" and hit the brakes!
                    Rover.throttle = 0
                    # Set brake to stored brake value
                    Rover.brake = Rover.brake_set
                    Rover.steer = 0
                    Rover.mode = 'stop'
        

        # If we're already in "stop" mode then make different decisions
        elif Rover.mode == 'stop':
            # If we're in stop mode but still moving keep braking
            if Rover.vel > 0.2:
                Rover.throttle = 0
                Rover.brake = Rover.brake_set
                Rover.steer = 0
            # If we're not moving (vel < 0.2) then do something else
            elif Rover.vel <= 0.2:
                # Now we're stopped and we have vision data to see if there's a path forward
                if len(Rover.nav_angles) < Rover.go_forward:
                    Rover.throttle = 0
                    # Release the brake to allow turning
                    Rover.brake = 0
                    # Turn range is +/- 15 degrees, when stopped the next line will induce 4-wheel turning
                    Rover.steer = 15 # Could be more clever here about which way to turn
                # If we're stopped but see sufficient navigable terrain in front then go!
                if len(Rover.nav_angles) >= Rover.go_forward:
                    # Set throttle back to stored value
                    Rover.throttle = Rover.throttle_set
                    # Release the brake
                    Rover.brake = 0
                    # Set steer to mean angle
                    Rover.steer = np.clip(np.mean(Rover.nav_angles * 180/np.pi), -15, 15)
                    Rover.mode = 'forward'
        
        elif Rover.mode == 'stuck':
            Rover.send_pickup = False
            # if 1 sec passed go back to previous mode
            if Rover.total_time - Rover.stuck_time > 1:
                # Set throttle back to stored value
                Rover.throttle = Rover.throttle_set
                # Release the brake
                Rover.brake = 0
                Rover.mode = 'forward' 
            # Now we're stopped and we have vision data to see if there's a path forward
            else:
                Rover.throttle = 0
                # Release the brake to allow turning
                Rover.brake = 0
                Rover.steer = 15
    # Just to make the rover do something 
    # even if no modifications have been made to the code
    else:
        Rover.send_pickup = False
        Rover.throttle = Rover.throttle_set
        Rover.steer = 0
        Rover.brake = 0
        
    # If in a state where want to pickup a rock send pickup command
    #if Rover.near_sample and Rover.vel == 0 and not Rover.picking_up:
        #Rover.send_pickup = True
    
    return Rover

