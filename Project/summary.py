import numpy as np
import sys
from math import *
import matplotlib.pyplot as plt
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

    # play stream (looping from beginning of file to the end)
    ######################Video################
    

    #--------------------Video---------------------------
    frames, col, row = 13500, 480, 270
    print("Reading the Videofile")
    image = np.fromfile(file_name_video, dtype ='uint8')
    image= np.reshape(image,(13500,270,480))
    frame1 = np.zeros(shape = (row,col,3), dtype = np.uint8)
    frame2 = np.zeros(shape = (row,col,3), dtype = np.uint8)
    final_entropy = np.zeros(shape = (4500,1), dtype = np.float64)
    tot_sum=row*col
    print("Calculating Heuristics")
    for i in range(0,frames-3,3):
        frame1=np.dstack((image[i+2][:][:],image[i+1][:][:],image[i][:][:]))
        frame1gray = cv2.cvtColor(frame1,cv2.COLOR_BGR2GRAY)
        frame2=np.dstack((image[i+5][:][:],image[i+4][:][:],image[i+3][:][:]))
        frame2gray = cv2.cvtColor(frame2,cv2.COLOR_BGR2GRAY)
        framediff = cv2.absdiff(frame2gray,frame1gray)
        hist = cv2.calcHist([framediff],[0],None,[256],[0,256])
        probs = hist/tot_sum
        sum_entropy = 0
        for j in range(255):
            if probs[j]==0:
                pass
            else:
                sum_entropy = sum_entropy + probs[j]*log(probs[j],2)
        final_entropy[i/3] = abs(sum_entropy)
    final_entropy= final_entropy/max(final_entropy)
    mean_full = np.mean(final_entropy)
    new_final_entropy = final_entropy-mean_full
    new_final_entropy[new_final_entropy < 0] =0
    cdf_array = np.zeros(shape=(90,1),dtype=np.float)
    for i in range(0,4500,50):
        cdf_array[i/50] = np.sum(new_final_entropy[i:i+49])
    
    new_cdf= np.zeros(shape=(90,1),dtype=np.float)
    new_cdf= np.argsort(cdf_array,axis=0)[::-1]
    selected_cdf = np.sort(new_cdf[0:28],axis=0)
    flag = 0
    
    #----------Video Player---------------------
    print("Launching Summary Video")
    cv2.namedWindow('Video', cv2.WINDOW_AUTOSIZE) 
    start_time = time.time()
    for i in range(0,28):
        k=selected_cdf[i]
        wf.setpos(chunk*50*k)
        data = wf.readframes(chunk)
        for j in range(50*k,50*k+49):
            Display_image = np.dstack((image[3*j+2][:][:],image[3*j+1][:][:],image[3*j][:][:]))
            cv2.imshow('Video', Display_image)
            stream.write(data)
            data = wf.readframes(chunk)
            q = cv2.waitKey(1)
            if (q == ord('s')):
                flag=1
                break
            elif(q == ord('p')):
                while(cv2.waitKey(1) != ord('p')):

                    if (cv2.waitKey(1) == ord('s')):
                        flag = 1
                        break
                    else:
                        pass

            else:
                pass

        if(flag == 1):
            break
            
    cv2.destroyAllWindows()
    stream.close()    
    p.terminate()
    elapsed_time = time.time() - start_time
    print "Elapsed Time : %d seconds" % (elapsed_time)

if __name__ == "__main__":
    sys.exit(int(main() or 0))
