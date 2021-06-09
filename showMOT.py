import cv2
import numpy as np
import os

imgpath = './data/track4/img1/'
annopath = './data/track4/gt/gt.txt'

anns = np.loadtxt(annopath, dtype=np.float32, delimiter=',')
anns = anns[np.argsort(anns[:, 0])]

images = sorted(os.listdir(imgpath))
image_filenames = [imgpath+image for image in images if 'PNG' in image]

for idx, iimg in enumerate(image_filenames):
    image = cv2.imread(iimg)
    Anno_info = anns[anns[:, 0]==idx+1]
    print(iimg)
    print(Anno_info)
    for i in range(len(Anno_info)):
        annotation = Anno_info[i]
        fid, tid, x, y, w, h, _, _, _ = annotation

        # 参数为(图像，左上角坐标，右下角坐标，边框线条颜色，线条宽度)
        # 注意这里坐标必须为整数，还有一点要注意的是opencv读出的图片通道为BGR，所以选择颜色的时候也要注意
        image = cv2.rectangle(
            image, (int(x), int(y)), (int(x+w), int(y+h)),
            (0, 255, 255), 2
        )
        image = cv2.putText(
            image, str(tid),
            (int(x), int(y)), cv2.FONT_HERSHEY_SIMPLEX, 0.5,
            (255, 0, 255), 2
        )
        # 参数为(显示的图片名称，要显示的图片)  必须加上图片名称，不然会报错
        #         cv2.imwrite('demo{}.png'.format(str(Img_id)), image)
    cv2.imshow('demo{}.png'.format(str(idx)), image)
    cv2.waitKey(500)
    cv2.destroyAllWindows()