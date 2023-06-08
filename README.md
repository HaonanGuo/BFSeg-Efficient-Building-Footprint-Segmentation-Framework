# BFSeg-Efficient-Building-Footprint-Segmentation-Framework
The manuscript is in the peer review process and thus only the evaluation code is avalible~<br/>
The training code will be released once it is accepted~ Coming Soon~

Dataset
----
[1.WHU building](https://study.rsgis.whu.edu.cn/pages/download/building_dataset.html)<br/>
[2.DeepGlobe](http://deepglobe.org/index.html)  <br/>
[3.Typical City Building Dataset](https://www.scidb.cn/en/detail?dataSetId=806674532768153600)  <br/>


The code
----
### Requirements
* torch
* torchvision
* pillow
* cv2

### Usage
Clone the repository:git clone https://github.com/HaonanGuo/BFSeg-Efficient-Building-Footprint-Segmentation-Framework.git<br/>

->Run [s2_Eval_BFSeg.py](https://github.com/HaonanGuo/BFSeg-Efficient-Building-Footprint-Segmentation-Framework/blob/main/s2_Eval_BFSeg.py) to evaluate the performance of BFSeg

Dont forget to download the model weights before evaluation:
| Dataset | WHU     | DeepGlobe     | Typical City     |
| :--------: | :--------: | :--------: | :--------: |
| Backbone | [ConvNext](https://drive.google.com/file/d/1olaP-AIywg1s6qqzeT-R5_CEse-veN5c/view?usp=drive_link) | [ConvNext](https://drive.google.com/file/d/1lQ76lw-hHrP2uEJN8Ht3Pwubel0HKA2h/view?usp=drive_link) | [ConvNext](https://drive.google.com/file/d/18SO-27LqE9zQ0X1lSr_rLqZbZRMuSLRh/view?usp=drive_link) |
| Backbone | [SwinTrans](https://drive.google.com/file/d/1s_lznG_56zIhiqO1Ei4p3IFqI5o-6k8z/view?usp=drive_link) | [SwinTrans](https://drive.google.com/file/d/1X02anXyT6oysmLRIo6ARNp7fc73dz1UY/view?usp=drive_link) | [SwinTrans](https://drive.google.com/file/d/1XFxUOKxRgcjuHdCsRAbZqyuY2FEooTHE/view?usp=drive_link) |


Help
----
Any question? Please contact us with: haonan.guo@whu.edu.cn
