import pandas as pd
import os

def resetID():
    gt_filepath = 'data/track3/gt/gt.txt'
    img_filepath = 'data/track3/img1'

    img_data = os.listdir(img_filepath)


    data = pd.read_csv(gt_filepath, header=None)
    min_id = min(data[0])
    print(min_id)
    if min_id == 1:
    #     return
    # else:
        data[0] = data[0]-min_id+1
        print(data)
        # reset gt.txt id
        data.to_csv('gt.txt', index=None, header=None)

        # reset image id
        for i in sorted(img_data):
            img_id = i.split(".")[0]
            print(int(img_id))
            old_name = os.path.join(
                img_filepath,
                i
            )
            new_name = os.path.join(
                img_filepath,
                "{:06d}.PNG".format(int(img_id) - int(min_id))
            )

            os.rename(old_name, new_name)
        return

if __name__ == '__main__':
    resetID()
