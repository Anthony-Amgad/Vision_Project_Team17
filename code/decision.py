import numpy as np
import time

# This is where you can build a decision tree for determining throttle, brake and steer 
# commands based on the output of the perception_step() function
def decision_step(Rover):

    if Rover.vel <= 0.1 and (Rover.total_time - Rover.stuck_time > 4) and Rover.mode != 'lockedin':
        # Set mode to "stuck" and hit the brakes
        Rover.stuck = True
        Rover.throttle = 0
        # Set brake to stored brake value
        Rover.brake = Rover.brake_set
        Rover.steer = 0
        Rover.stuck_time = Rover.total_time

    
    if ((len(Rover.samples_angles) > 5) or (len(Rover.samples_angles2) > 10)) and (not Rover.stuck) and Rover.mode != 'found' and Rover.mode != 'lockedin' and Rover.mode != 'pick' and Rover.mode != 'pickup' and Rover.mode != 'stop':
        Rover.mode = 'found'
        print('FOUND')
        Rover.stuck_time = Rover.total_time

    elif Rover.mode == 'found' and (len(Rover.samples_angles3) > 0):
        Rover.send_pickup = False
        Rover.brake = 5
        Rover.throttle = 0
        if Rover.vel == 0 and (Rover.pitch < 0.5 or Rover.pitch > 359.5) and (Rover.roll < 0.5 or Rover.roll > 359.5):
            Rover.mode = 'lockedin'
    
    
    elif Rover.mode == 'lockedin':
        if  (len(Rover.samples_angles3) > 0):
            mangle = np.mean(Rover.samples_angles3 * 180/np.pi)

            if (mangle < 15) and (mangle > -15):
                Rover.mode = 'pick'
            else:
                Rover.brake = 0
                Rover.steer = np.clip(mangle, -15, 15)
                Rover.stuck_time = Rover.total_time
        
        else:
            Rover.brake = 0
            Rover.steer = 15
            Rover.stuck_time = Rover.total_time

    
    elif Rover.mode == 'pick' and (len(Rover.samples_angles3) > 0) and (not Rover.stuck):     
        mangle = np.mean(Rover.samples_angles3 * 180/np.pi)
        if Rover.vel < Rover.max_vel:
            # Set throttle value to throttle setting
            Rover.throttle = Rover.throttle_set
            Rover.brake = 0
            # Set steering to average angle clipped to the range +/- 15
            Rover.steer = np.clip(mangle, -15, 15)
        else: # Else coast
            Rover.throttle = 0
        
        if np.mean(Rover.samples_dists3) < 10:
            # Set mode to "stop" and hit the brakes!
            Rover.throttle = 0
            # Set brake to stored brake value
            Rover.brake = Rover.brake_set
            Rover.steer = 0
            Rover.mode = 'pickup'

    elif Rover.mode == 'pick' and (len(Rover.samples_angles3) == 0) and (not Rover.stuck):
        Rover.mode = 'found'
            

    elif Rover.mode == 'pickup':
        if Rover.vel > 0.05:
            Rover.brake = Rover.brake_set
        elif not Rover.picking_up:
            Rover.stuck_time = Rover.total_time
            Rover.send_pickup = True
            Rover.mode = 'stop'

    elif Rover.mode == 'pick' and Rover.stuck:
        # if 0.3 sec passed go back to previous mode
        if Rover.total_time - Rover.stuck_time > 0.3:
            # Set throttle back to stored value
            Rover.throttle = Rover.throttle_set
            # Release the brake
            Rover.brake = 0
            Rover.stuck = False
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
        if Rover.mode == 'forward' and (not Rover.stuck): 
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
        elif Rover.mode == 'stop' and (not Rover.stuck):
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
                if (len(Rover.nav_angles) >= Rover.go_forward) and (not Rover.picking_up) and (not Rover.send_pickup):
                    # Set throttle back to stored value
                    Rover.throttle = Rover.throttle_set
                    # Release the brake
                    Rover.brake = 0
                    # Set steer to mean angle
                    Rover.steer = np.clip(np.mean(Rover.nav_angles * 180/np.pi), -15, 15)
                    Rover.mode = 'forward'
        
        elif Rover.stuck:
            Rover.send_pickup = False
            # if 1 sec passed go back to previous mode
            if Rover.total_time - Rover.stuck_time > 1:
                # Set throttle back to stored value
                Rover.throttle = Rover.throttle_set
                # Release the brake
                Rover.brake = 0
                Rover.stuck = False 
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

    print("##################### "+ Rover.mode + " ########### " + str(Rover.stuck))
    
    return Rover

