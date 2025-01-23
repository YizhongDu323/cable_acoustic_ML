from robodk import *      # RoboDK API
from robolink import *    # Robot toolbox

# Link to RoboDK
RDK = Robolink()

# Notify user:
print('To edit this program:\nright click on the Python program, then, select "Edit Python script"')

# Get the robot and targets by their names
robot = RDK.Item('Fanuc LR Mate 200iD/7L')  # Ensure this is the correct robot name in RoboDK
target1 = RDK.Item('Target 1')                # Ensure 'Target1' exists in your RoboDK station
target2 = RDK.Item('Target 2')                # Ensure 'Target2' exists in your RoboDK station

# Check if the robot is valid (i.e., exists and is selected)
if robot.Valid():
    print('Item selected: ' + robot.Name())
    print('Item position: ' + repr(robot.Pose()))  # This prints the current position of the robot

    # Make sure that the targets are valid as well
    if target1.Valid() and target2.Valid():
        # Move the robot to target1 and target2 sequentially
        while True:
            robot.MoveJ(target1)  # Move to target1
            robot.MoveJ(target2)  # Move to target2
    else:
        print('Error: One or more targets are not valid.')
else:
    print('Error: Robot not found or invalid.')
