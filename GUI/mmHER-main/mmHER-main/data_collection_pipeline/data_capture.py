from steaming import adcCapThread
import time
import sys

if __name__ == '__main__':
    a = adcCapThread(1, "adc")
    a.start()
    counter = 0
    t_start = time.time()
    
    f = open("data_capture.bin", "wb") 
    duration = int(sys.argv[1])  # now in seconds

    while True:
        readItem, itemNum, lostPacketFlag = a.getFrame()
        if itemNum > 0:
            counter += 1
            f.write(readItem.tobytes())
        elif itemNum == -1:
            print(readItem)
        elif itemNum == -2:       
            time.sleep(0.04)

        elapsed_time = time.time() - t_start
        if counter/20 == duration:
            a.whileSign = False 
            f.close()
            break

    print('Done')
    print('\a')

# from steaming import adcCapThread
# import time
# import sys

# if __name__=='__main__':
#     a = adcCapThread(1,"adc")
#     a.start()
#     counter = 0
#     t = time.time()
#     time_min = 0
    
#     f = open(str(t).split(".")[0]+".bin", "wb")
#     duration=int(sys.argv[1])
    
#     while True:
#         readItem,itemNum,lostPacketFlag=a.getFrame()
#         if itemNum>0:
#             counter+=1
#             f.write(readItem.tobytes())
#             if counter == 1200:
#                 counter = 0
#                 time_min+=1        
            
#         elif itemNum==-1:
#             print(readItem)
#         elif itemNum==-2:       
#             time.sleep(0.04)
#         if time_min>=duration:    
#            a.whileSign = False 
#            f.close()
#            break
#     print('Done')
#     print('\a')
