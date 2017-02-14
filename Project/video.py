import numpy as np
import sys
import cv2
import time
import pyaudio
import wave

def main():

    file_name_video = sys.argv[1]
    file_name_audio  = sys.argv[2]
    #####################3Audio##############333
    # length of data to read.
    chunk = 1600

    # open the file for reading.
    print("Reading the Audiofile")
    wf = wave.open(file_name_audio, 'rb')

    # create an audio object
    p = pyaudio.PyAudio()

    # open stream based on the wave object which has been input.
    stream = p.open(format =
                    p.get_format_from_width(wf.getsampwidth()),
                    channels = wf.getnchannels(),
                    rate = wf.getframerate(),
                    output = True)

    ######################Video################
    

    #-------------Video---------------------------------------------------
    frames, col, row = 13500, 480, 270
    print("Reading the Videofile")
    image = np.fromfile(file_name_video, dtype ='uint8')
    image= np.reshape(image,(13500,270,480))
    Display_image = np.zeros(shape = (row,col,3), dtype = np.uint8)
    
    print("Launching Video")
    cv2.namedWindow('Video', cv2.WINDOW_AUTOSIZE)
    flag = 0
    wf.setpos(0)
    data = wf.readframes(chunk)
    i = 0
    while(i < frames):
        Display_image = np.dstack((image[i+2][:][:],image[i+1][:][:],image[i][:][:]))
        cv2.imshow('Video', Display_image)
        stream.write(data)
        data = wf.readframes(chunk)
        q = cv2.waitKey(1)
        if (q == ord('s')):
            break
        elif(q == ord('p')):
            while(cv2.waitKey(1) != ord('p')):

                if (cv2.waitKey(1) == ord('s')):
                    flag = 1
                    break
                else:
                    pass
        elif(q == ord('r')):
            while(cv2.waitKey(1) != ord('r')):
                if(i<3):
                    i=frames-3
                i -= 3
                Display_image = np.dstack((image[i+2][:][:],image[i+1][:][:],image[i][:][:]))
                cv2.imshow('Video', Display_image)
            wf.setpos(chunk*(i/3))
        elif(q == ord('f')):
            while(cv2.waitKey(1) != ord('f')):
                if (i>(frames-6)):
                    i=0
                i += 3
                Display_image = np.dstack((image[i+2][:][:],image[i+1][:][:],image[i][:][:]))
                cv2.imshow('Video', Display_image)
            wf.setpos(chunk*(i/3))
        else:
            pass

        if(flag == 1):
            break

        i += 3

    # cleanup stuff.
    stream.close()    
    p.terminate()   
    cv2.destroyAllWindows()

if __name__ == "__main__":
    sys.exit(int(main() or 0))