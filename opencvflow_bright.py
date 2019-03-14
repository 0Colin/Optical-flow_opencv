import cv2
import numpy as np

# add your video absolute path
pathv = 'C:\\your path\\v_GolfSwing_g17_c05.avi'

#or you can just ./v_GolfSwing_g17_c05.avi,but may report an error when you run the program
cap = cv2.VideoCapture(pathv)
ret, frame1 = cap.read()
prvs = cv2.cvtColor(frame1,cv2.COLOR_BGR2GRAY)

# optical flow visualization 光流可视化
    # 由于frame的数据类型为np.uint8 即 usigned char, 最大存储值为255, 如果赋值为256, 结果为 0,
    # 也就是说及时赋值很大, 也会被截断
    # 对于 饱和度s 和亮度v 而言, 最大值是255,
    #  s = 255 色相最饱和, v = 255, 图片最亮
    # 而对与颜色而言, opencv3中, (0, 180) 就会把所有颜色域过一遍, 所以这里下面就算角度时会除以 2

# np.zeros_like(): Return an array of zeros with the same shape and type as a given array.
hsv = np.zeros_like(frame1)
# hsv[...,1] = 255
# while(1):
while cap.isOpened():
    ret, frame2 = cap.read()
    # print (frame2)
    next = cv2.cvtColor(frame2,cv2.COLOR_BGR2GRAY)
    flow = cv2.calcOpticalFlowFarneback(prvs,next, None, 0.5, 3, 15, 3, 5, 1.2, 0)  #need to try
    # mag, ang = cv2.cartToPolar(flow[...,0], flow[...,1])
    mag, ang = cv2.cartToPolar(flow[...,0], flow[...,1])
    hsv[...,0] = ang*180/np.pi/2
    #  brief Normalizes the norm or value range of an array
    # norm_type = cv2.NORM_MINMAX, 即将值标准化到(0, 255)
    hsv[..., 1] = cv2.normalize(mag, None, 0, 255, cv2.NORM_MINMAX)
    
    # 亮度为255
    hsv[..., 2] = 255

    # hsv[...,2] = cv2.normalize(mag,None,0,255,cv2.NORM_MINMAX)
    rgb = cv2.cvtColor(hsv,cv2.COLOR_HSV2BGR)
    cv2.imshow('frame2',rgb)
    k = cv2.waitKey(30) & 0xff
    if k == 27:
    # if k == ord('q'):
        break
    elif k == ord('s'):
        # print ("i am in")
        cv2.imwrite('opticalfb.png',frame2)
        cv2.imwrite('opticalhsv.png',rgb)
    prvs = next

cap.release()
cv2.destroyAllWindows()


