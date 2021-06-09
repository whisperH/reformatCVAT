from xml.dom.minidom import parse
import numpy as np
import os
import shutil
dataset = 'track8'
annos_path = './data/'+dataset+'/annotations.xml'
gt_path = './data/'+dataset+'/gt/'

if os.path.exists(gt_path):
    shutil.rmtree(gt_path)
os.makedirs(gt_path)

def readXML():
    domTree = parse(annos_path)
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
    np.savetxt(gt_path+'gt.txt', gt, fmt='%.04f', delimiter=',')

if __name__ == '__main__':
    readXML()
