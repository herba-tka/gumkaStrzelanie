import cv2
import numpy as np

# define a video capture object
vid = cv2.VideoCapture(0)

lastAVG = np.zeros((480, 640), dtype=np.float)

alpha = 0.05

while (True):

    # Capture the video frame
    # by frame
    ret, frame = vid.read()
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

    lastAVG = (lastAVG*(1-alpha) + gray.astype(np.float)*alpha)
    # Difference between static background
    # and current frame(which is GaussianBlur)
    diff_frame = cv2.absdiff(lastAVG.astype(np.uint8), gray)

    # If change in between static background and
    # current frame is greater than 30 it will show white color(255)
    thresh_frame = cv2.threshold(diff_frame, 30, 255, cv2.THRESH_BINARY)[1]
    thresh_frame = cv2.dilate(thresh_frame, None, iterations=2)

    # Finding contour of moving object
    cnts, _ = cv2.findContours(thresh_frame.copy(),
                               cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

    motions = []

    for contour in cnts:
        if cv2.contourArea(contour) < 2000:
            continue

        (x, y, w, h) = cv2.boundingRect(contour)
        # making green rectangle around the moving object
        cv2.rectangle(frame, (x, y), (x + w, y + h), (0,0,255), 3)

        motions.append(np.array([x+w/2, y+h/2], dtype=np.float))

    if len(motions) > 0:
        print("motions", len(motions))
        motion = np.mean(motions, axis=0)

        x,y = motion[0], motion[1]
        cv2.rectangle(frame,(int(x),int(y)), (min(480, int(x+30)), min(640, int(y+30))), (0,255,0), -1)




    # Display the resulting frame
    cv2.imshow('averaged', thresh_frame.astype(np.uint8))
    cv2.imshow('frame', frame)
    # cv2.imshow('averaged', frame)


    # the 'q' button is set as the
    # quitting button you may use any
    # desired button of your choice
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break


# After the loop release the cap object
vid.release()
# Destroy all the windows
#%%
cv2.destroyAllWindows()
#%%