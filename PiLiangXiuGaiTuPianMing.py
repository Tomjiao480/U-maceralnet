import os

class BatchRename():

    def __init__(self):
        self.path = 'G:/Code/Matalab/ColorTransfer/Save_Images'
        self.save_path='G:/Code/Matalab/ColorTransfer/Save_Images_2'
    def rename(self):
        filelist = os.listdir(self.path)
        total_num = len(filelist)
        i = 1
        for item in filelist:
            print(item)
            if item.endswith('.jpg'):
                src = os.path.join(os.path.abspath(self.path), item)
                dst = os.path.join(os.path.abspath(self.save_path), ''+str(i) + '.jpg')
                try:
                    os.rename(src, dst)
                    print ('converting %s to %s ...' % (src, dst))
                    i = i + 1
                except:
                    continue
        print ('total %d to rename & converted %d jpgs' % (total_num, i))

if __name__ == '__main__':
    demo = BatchRename()
    demo.rename()

