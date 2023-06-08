import torch
import torch.nn.functional as F
from tqdm import tqdm
import numpy as np
from tqdm import tqdm
from PIL import Image
import os

class BuildingDataset_iter(Dataset):
    def __init__(self, type,base_dir='../WHU_BUILDING'):
        assert type in ['train','val','test']
        self.type=type
        self.imgs_dir = os.path.join(base_dir,type,'image')
        self.masks_dir = os.path.join(base_dir,type,'label')
        self.mean,self.std,self.shuffix=[0.43782742, 0.44557303, 0.41160695],[0.19686149, 0.18481555, 0.19296625],'.tif'
        self.ids = [splitext(file)[0] for file in listdir(self.imgs_dir) if not file.startswith('.')]
        logging.info(f'BUILDING dataset with {len(self.ids)} examples')
    def __len__(self):
         return len(self.ids)
    def __getitem__(self, i):

        idx = self.ids[i]
        mask_file = os.path.join(self.masks_dir , idx + self.shuffix)
        img_file = os.path.join(self.imgs_dir ,idx + self.shuffix)
        mask = Image.open(mask_file)
        img = Image.open(img_file)
        img,mask=np.array(img),(np.array(mask)>0).astype(np.uint8)
        img,mask=transF.to_tensor(img.copy()),(transF.to_tensor(mask.copy())>0).int()
        img=transF.normalize(img,self.mean,self.std)
        return {
            'image': img.squeeze().float(),
            'mask': mask.squeeze().float(),
            'name':os.path.join(self.imgs_dir,idx)
        }
        
def fast_hist(label_pred, label_true, num_classes):
    mask = (label_true >= 0) & (label_true < num_classes)
    hist = np.bincount(
        num_classes * label_true[mask].astype(int) +
        label_pred[mask], minlength=num_classes ** 2).reshape(num_classes, num_classes)
    return hist

def eval_net(net, loader,pre_test=False,savename=None):
    """Evaluation without the densecrf with the dice coefficient"""
    net.eval()
    hist=0

    for num,batch in enumerate(loader):
        if num%20==0:
            print(num,' in ',len(loader))
        imgs, true_masks,name = batch['image'].cuda(), batch['mask'].cuda(),batch['name']
        b,c,h,w=imgs.shape
        with torch.no_grad():
            netpred = net(imgs,true_masks)
            mask_pred=netpred[0]
            pred = torch.sigmoid(mask_pred)> 0.5

        hist+=fast_hist(pred.flatten().cpu().detach().int().numpy(),true_masks.flatten().cpu().int().numpy(),num_classes)
        if savename is not None:
            Dataset,read_name=savename
            savedir = os.path.join('./predict' ,Dataset)
            if os.path.exists(os.path.join('./predict' ,Dataset)) is False:
                os.makedirs(os.path.join('./predict' ,Dataset))
            tmp = (pred.squeeze() * 2 + true_masks.squeeze()).cpu().numpy()
            out = np.zeros((b,h, w, 3))
            out[tmp == 1,:] = [0, 0, 255]
            out[tmp == 2,:] = [255, 0, 0]
            out[tmp == 3,:] = [0, 255, 0]
            out=out.astype(np.uint8)
            for i in range(len(name)):
                Image.fromarray(out[i]).save(os.path.join(savedir,name[i].split('/')[-1]+'_'+read_name[:-5]+'.png'))

    per_class_iou= np.diag(hist) / (hist.sum(1) + hist.sum(0) - np.diag(hist))
    print('Binary IOU:',per_class_iou[-1]*100)
    print(hist)
    return per_class_iou[-1]

if __name__=='__main__':
    net=torch.load('./model.pth')
    eval(net,BuildingDataset_iter('test'))
