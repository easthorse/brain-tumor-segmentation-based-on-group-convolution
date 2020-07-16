import os

if __name__ == '__main__':
    compiler = '/home/ubuntu/anaconda3/envs/zym/bin/python'
    test_py = '/media/ubuntu/1F4F46694A801839/zym/BraTS-DMFNet-master/test.py'

    ckpt_folder = '/media/ubuntu/1F4F46694A801839/zym/BraTS-DMFNet-master/ckpts/MFNet_GDL_all_semix'

    # pths = os.listdir(ckpt_folder)
    # pths = range(len([file for file in pths if file.endswith('.pth')]))[]

    prefix = 'model_epoch_'
    posfix = '.pth'

    for idx in range(400, 500, 1):
        pth = prefix + str(idx) + posfix

        test_cmd = compiler + ' ' + test_py + ' --restore ' + pth
        print(test_cmd)

        os.system(test_cmd)
