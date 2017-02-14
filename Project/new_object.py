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
    file_name_frame  = sys.argv[3]
    height = int(sys.argv[4])
    width = int(sys.argv[5])
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
    frames, col, row = 13500, 480, 270

    # print("Starting the code")
    print("Reading the Videofile")
    image = np.fromfile(file_name_video, dtype ='uint8')
    testimage=np.fromfile(file_name_frame, dtype ='uint8')

    #Reshaping Videofile
    image= np.reshape(image,(13500,270,480))
    
    #Reshaping the test frame
    testimage1=np.reshape(testimage,(3,height,width))
    stack = np.dstack((testimage1[2][:][:],testimage1[1][:][:],testimage1[0][:][:]))

    #Resizing the test image
    testimage = cv2.resize(stack, (col,row))
    frame1 = np.zeros(shape = (row,col,3), dtype = np.uint8)
    
    #Image to gray
    testimagegray = cv2.cvtColor(testimage,cv2.COLOR_BGR2GRAY)
    final_entropy = np.zeros(shape = (4500,1), dtype = np.float64)
    tot_sum=row*col
    
    print("Analysing the Videofile")
    for i in range(0,frames,3):
        frame1=np.dstack((image[i+2][:][:],image[i+1][:][:],image[i][:][:]))
        frame1gray = cv2.cvtColor(frame1,cv2.COLOR_BGR2GRAY)
        framediff = cv2.absdiff(testimagegray,frame1gray)
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
    index = np.argmin(final_entropy)
    minima = np.min(final_entropy)
    mean_line = (np.ones(final_entropy.shape))*mean_full
    new_final_entropy = np.zeros(final_entropy.shape)
 
    #Smoothing Average 
    for j in range(10):
        for i in range(0,final_entropy.shape[0]):
            if i==0:
                new_final_entropy[i] = (final_entropy[i] + final_entropy[i+1])/2
            elif i==final_entropy.shape[0]-1:
                new_final_entropy[i] = (final_entropy[i] + final_entropy[i-1])/2
            else:
                new_final_entropy[i] = (final_entropy[i-1] + final_entropy[i] + final_entropy[i+1])/3
        final_entropy = new_final_entropy

    #Segmenting and Indexing
    # Centre Left
    count=0
    i=index
    while i <=index:
        slope = new_final_entropy[i-1] - new_final_entropy[i]
        if slope>0:
            if count >75:
                start_frame=index-count
                break
            elif i<=0:
                start_frame=i
                break
            else:
                i-=1
                count+=1
            
        elif (slope < 0) and (new_final_entropy[i]>mean_full):
            start_frame=i
            break
        
        else:
            if count >75:
                start_frame=index-count
                break
            elif i<=0:
                start_frame=i
                break
            else:
                i-=1
                count+=1

    #Centre Right
    i=index
    count=0
    while i >=index:
        slope = new_final_entropy[i+1] - new_final_entropy[i]
        if slope>0:
            if count >75:
                end_frame=index+count
                break
            elif i>=4500:
                end_frame=i
                break
            else:
                i+=1
                count+=1
            
        elif (slope < 0) and (new_final_entropy[i]>mean_full):
            end_frame=i
            break

        else:
            if count >75:
                end_frame=index+count
                break
            elif i>=4500:
                end_frame=i
                break
            else:
                i+=1
                count+=1

    

    print("Playing the Section")
    flag=0;
    cv2.namedWindow('Video', cv2.WINDOW_AUTOSIZE)
    wf.setpos(chunk*start_frame)
    data=wf.readframes(chunk)
    for i in range(start_frame*3,end_frame*3+3,3):
        Display_image = np.dstack((image[i+2][:][:],image[i+1][:][:],image[i][:][:]))
        cv2.imshow('Video',Display_image)
        stream.write(data)
        data=wf.readframes(chunk)
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
                i -= 3
                Display_image = np.dstack((image[i+2][:][:],image[i+1][:][:],image[i][:][:]))
                cv2.imshow('Video', Display_image)
            wf.setpos(chunk*(i/3))
        elif(q == ord('f')):
            while(cv2.waitKey(1) != ord('f')):
                i += 3
                Display_image = np.dstack((image[i+2][:][:],image[i+1][:][:],image[i][:][:]))
                cv2.imshow('Video', Display_image)
            wf.setpos(chunk*(i/3))
        else:
            pass

        if(flag == 1):
            break


    cv2.destroyAllWindows()
    stream.close()    
    p.terminate()

if __name__ == "__main__":
    sys.exit(int(main() or 0))
