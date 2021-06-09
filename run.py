from xml.dom.minidom import parse
import configparser
import numpy as np
import os
import shutil
import pandas as pd
import cv2
from parseXML import *

dataset = 'rest'
dataset_path = '/media/whisper/Other/data/individual_behavior'

# old_img_file = os.path.join(dataset_path, dataset, 'images')
img_file = os.path.join(dataset_path, dataset, 'img1')
# 如果存儲圖片的文件夾不是img1而是其他的，則需要重命名
# if not os.path.exists(img_file):
#     os.rename(old_img_file, img_file)

# 如果mot真值gt文件夾不存在，則需要先創建新的文件夾
mot_gt_filepath = os.path.join(dataset_path, dataset, 'gt')

if os.path.exists(mot_gt_filepath):
    shutil.rmtree(mot_gt_filepath)
os.makedirs(mot_gt_filepath)

mot_gt_file = os.path.join(mot_gt_filepath, 'gt.txt')
mot_label_file = os.path.join(dataset_path, dataset, 'seqinfo.ini')

cvat_anno_file = os.path.join(dataset_path, dataset, 'annotations.xml')

class FormatCVAT(object):
    def __init__(self):
        self.tree = read_xml(cvat_anno_file)
        self.img_data = sorted(os.listdir(img_file))

    def writeConfig(self):
        seqLength = find_nodes(self.tree, 'meta/task/size')[0].text
        image = cv2.imread(os.path.join(
            img_file, self.img_data[0]
        ))
        imHeight = image.shape[0]
        imWidth = image.shape[1]

        # 创建管理对象
        conf = configparser.ConfigParser()

        # 添加一个select
        conf.add_section("Sequence")

        # 往select添加key和value
        conf.set("Sequence", "name", dataset)
        conf.set("Sequence", "imDir", "img1")
        conf.set("Sequence", "frameRate", "25")
        conf.set("Sequence", "seqLength", seqLength)
        conf.set("Sequence", "imWidth", str(imWidth))
        conf.set("Sequence", "imHeight", str(imHeight))
        conf.set("Sequence", "imExt", '.PNG')


        try:
            path = 'meta/task/labels/label/attributes/attribute/values'
            values = find_nodes(self.tree, path)
            labels = values[0].text.split()
            print(labels)
            conf.add_section("State")
            for idx, label in enumerate(labels):
                conf.set("State", label, str(idx))

            print(f"label file is saved in {mot_label_file}")
        except Exception as e:
            print("no need parse labels")

        conf.write(open(mot_label_file, 'w'))

    def CVAT2MOT(self):
        domTree = parse(cvat_anno_file)
        # 文档根元素
        rootNode = domTree.documentElement
        print(rootNode.nodeName)

        # 所有顾客
        tracks = rootNode.getElementsByTagName("track")
        print("****所有tracking information****")
        gt = []
        for track in tracks:
            track_id = track.getAttribute("id")
            print("fish ID:", track_id)
            boxes = track.getElementsByTagName("box")
            for ibox in boxes:
                frame_id = ibox.getAttribute("frame")
                occluded = ibox.getAttribute("occluded")
                xtl = ibox.getAttribute("xtl")
                ytl = ibox.getAttribute("ytl")
                xbr = ibox.getAttribute("xbr")
                ybr = ibox.getAttribute("ybr")

                state = ibox.getElementsByTagName("attribute")[0].childNodes[0].data

                w = float(xbr)-float(xtl)
                h = float(ybr)-float(ytl)
                igt = [
                    int(frame_id)+1, int(track_id) + 1,
                    float(xtl), float(ytl), w, h,
                    1, 1, int(occluded)
                ]
                gt.append(igt)
        gt = np.array(gt)
        flist = gt.T[:2, :][0]
        tlist = gt.T[:2, :][1]

        idx = np.lexsort((tlist, flist))
        gt = gt[idx, :]
        print(gt)
        np.savetxt(mot_gt_file, gt, fmt='%d', delimiter=',')

    def resetMOTID(self):
        data = pd.read_csv(mot_gt_file, header=None)
        min_id = min(data[0])
        print(min_id)
        if min_id == 1:
            print("no need to reset ID")
        else:

            data[0] = data[0]-min_id+1
            # reset gt.txt id
            # data.to_csv(mot_gt_file, index=None, header=None)
            np.savetxt(mot_gt_file, data.values, fmt='%d', delimiter=',')
            print("reset MOTID is ok")

    def resetImgID(self):
        # reset image id
        min_id = int(self.img_data[0].split(".")[0].split("_")[-1])
        if min_id == 1:
            print("no need rest image id")
            return
        else:
            for i in self.img_data:
                img_id = int(i.split(".")[0].split("_")[-1])
                print(int(img_id))
                new_id = img_id - min_id + 1
                old_name = os.path.join(
                    img_file,
                    i
                )
                new_name = os.path.join(
                    img_file,
                    "{:06d}.PNG".format(int(new_id))
                )

                os.rename(old_name, new_name)
            print("rest image id is ok")
            return

    def showMOT(self):
        anns = np.loadtxt(mot_gt_file, dtype=np.float32, delimiter=',')
        anns = anns[np.argsort(anns[:, 0])]

        for idx, iimg in enumerate(self.img_data):
            image = cv2.imread(os.path.join(
                img_file, iimg
            ))
            Anno_info = anns[anns[:, 0] == idx + 1]
            print(Anno_info)
            for i in range(len(Anno_info)):
                annotation = Anno_info[i]
                fid, tid, x, y, w, h, _, _, _ = annotation

                # 参数为(图像，左上角坐标，右下角坐标，边框线条颜色，线条宽度)
                # 注意这里坐标必须为整数，还有一点要注意的是opencv读出的图片通道为BGR，所以选择颜色的时候也要注意
                image = cv2.rectangle(
                    image, (int(x), int(y)), (int(x + w), int(y + h)),
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
            cv2.waitKey(1000)
            cv2.destroyAllWindows()
    def run(self):
        self.writeConfig()
        self.CVAT2MOT()
        self.resetMOTID()
        self.resetImgID()
        # showMOT()
if __name__ == '__main__':
    FormatCVAT().run()
