# brain-tumor-segmentation-based-on-group-convolution
Key Words : Magnetic Resonance Imaging, brain tumor segmentation, deep learning, group convolution, weighted mixed loss function


## Overview

  Brain tumor is a serious threat to human health. The invasive growth of brain tumor, when it occupies a certain space in the skull, will lead to increased intracranial pressure and compression of brain tissue, which will damage the central nerve and even threaten the patient’s life. Therefore, effective brain tumor diagnosis and timely treatment are of great significance to improving the patient’s quality of life and prolonging the patient’s life. Computer-assisted segmentation of brain tumor is necessary for the prognosis and treatment of patients. However, although brain-related research has made great progress, automatic identification of the contour information of tumor and effective segmentation of each subregion in MRI remain difficult due to the highly heterogeneous appearance, random location, and large difference in the number of voxels in each subregion of the tumor and the high degree of gray-scale similarity between the tumor tissue and neighboring normal brain tissue. Since 2012, with the development of deep learning and the improvement of related hardware performance, segmentation methods based on neural networks have gradually become the mainstream. In particular, 3D convolutional neural networks are widely used in the field of brain tumor segmentation because of their advantages of sufficient spatial feature extraction and high segmentation effect. Nonetheless, their large memory consumption and high requirements on hardware resources usually require making a compromise in the network structure that adapts to the given memory budget at the expense of accuracy or training speed. To address such problems, we propose a lightweight segmentation algorithm in this paper.

<div  align="center">  
 <img src="https://github.com/easthorse/brain-tumor-segmentation-based-on-group-convolution/blob/base/figure/Figure1.png"
     align=center/>
</div>

<center>Figure 1 MRI images in different modes（（a）FLAIR;（b）T1;（c）TIC;（d）T2) </center>


## Method 

  First, group convolution was used to replace conventional convolution for significantly reducing the parameters and improving segmentation accuracy because memory consumption is negatively correlated with batch size. A large batch size usually means enhanced convergence stability and training effect in 3D convolutional neural networks. 
  
<div  align="center">  
 <img src="https://github.com/easthorse/brain-tumor-segmentation-based-on-group-convolution/blob/base/figure/Figure4.png"
     align=center/>
</div>

<center> Figure 4 A structural diagram of group convolution and Multi-Fiber units ((a) schematic diagram of two consecutive convolutions; (b)schematic diagram of two group convolution layers with a number of groups of three; (c) architecture details of Multi-Fiber unit) </center>
  
  Then, multifiber and channel shuffle units were used to enhance the information fusion among the groups and compensate for the poor communication caused by group convolution. 
  <div  align="center">  
 <img src="https://github.com/easthorse/brain-tumor-segmentation-based-on-group-convolution/blob/base/figure/Figure5.png"
 <center>figure 5 A schematic diagram of the Channel Shuffle unit. ((a) schematic diagram of group convolution without channel shuffle unit; (b) implementation principle diagram of channel shuffle unit; (c) equivalent schematic diagram of group convolution added to channel shuffle unit)  </center>
     align=center/>
</div>

  Synchronized cross-GPU batch normalization was used to alleviate the poor training performance of 3D convolutional neural networks due to the small batch size and utilize the advantages of multigraphics collaborative computing. Aiming at the case in which the subregions have different difficulties in segmentation, a weighted mixed-loss function consisting of Dice and Jaccard losses was proposed to improve the segmentation accuracy of the subregions that are difficult to segment under the premise of maintaining the high precision of the easily segmented subregions and accelerate the model convergence speed. One of the most challenging parts of the task is to distinguish between small blood vessels in the tumor core and enhanced-tumor areas. This process is particularly difficult for the labels that may not have enhanced tumor at all. If neither the ground truth nor the prediction has an enhanced area, the Dice score of the enhancement area is 1. Conversely, in patients who did not have enhanced tumors in the ground truth, only a single false-positive voxel would result in a Dice score of 0. Hence, we postprocessed the prediction results, that is, we set a threshold for the number of voxels in the tumor-enhanced area. When the number of voxels in the tumor-enhanced area is less than the threshold, these voxels would be merged into the tumor core area, thereby improving the Dice score of the tumor-enhanced and tumor core areas.
  
<div  align="center">  
 <img src="https://github.com/easthorse/brain-tumor-segmentation-based-on-group-convolution/blob/base/figure/Figure2.png"
     align=center/>
</div>
<center>Figure 2 Schematic diagram of the complete network structure</center>
<div  align="center">  
 <img src="https://github.com/easthorse/brain-tumor-segmentation-based-on-group-convolution/blob/base/figure/Figure3.png"
     align=center/>
</div>

<center>Figure 3 Schematic diagram of MFS unit based on group convolution and residual structure
((a) MFS unit when channels numbers and size of the input layer are same as output layer’s ;(b) MFS unit when channels numbers and the size of the input layer are the same as the output layer’s )
</center>

## Result

  TTo verify the overall performance of the algorithm, we first conducted a fivefold cross-validation evaluation on the training set of the public brain tumor dataset BraTS2018. The average Dice scores of the proposed algorithm in the entire tumor, tumor core, and enhanced tumor areas can reach 89.52%, 82.74%, and 77.19%, respectively. For fairness, an experiment was also conducted on the BraTS2018 validation set. We used the trained network to segment the unlabeled samples for prediction, converted them into the corresponding format, and uploaded them to the BraTS online server. The segmentation results were provided by the server after calculation and analysis. The proposed algorithm achieves average Dice scores of 90.67%, 85.06%, and 80.41%. The parameters and floating point operations are 3.2 M and 20.51 G, respectively. Compared with the classic 3D U-Net, our algorithm shows higher average Dice scores by 2.14%, 13.29%, and 4.45%. Moreover, the parameters and floating point operations are reduced by 5 and 81 times, respectively. Compared with the state-of-the-art approach that won the first place in the 2018 Multimodal Brain Tumor Segmentation Challenge, the average Dice scores are reduced by only 0.01%, 0.96%, and 1.32%. Nevertheless, the parameters and floating point operations are reduced by 12 and 73 times, respectively, indicating a more practical value. Conclusion Aiming at the problems of large memory consumption and slow segmentation speed in the field of computer-aided brain tumor segmentation, an algorithm combining group convolution and channel shuffle unit is proposed. The punishment intensity of sparse class classification error to model is improved using the weighted mixed loss function to balance the training intensity of different segmentation difficulty categories effectively. The experimental results show that the algorithm can significantly reduce the computational cost while maintaining high accuracy and provide a powerful reference for clinicians in brain tumor segmentation.

Table 1 Comparison of various algorithms on the BraTS2018 validation set.

| Methods                    | Parameters(M) | FLOPs(G)  |  |  Dice   |            |          |     HD95        |          |
|:---------------------------:|:---------------:|:-----------:|:-----------:|:-----------:|:-----------:|----------|:----------:|:----------:|
|                            |               |           | ET        | WT        | TC        | ET       | WT       | TC       |
| **Ours**                   | **3.2**       | **20.51** | **80.41** | **90.67** | **85.06** | **2.51** | **4.13** | **5.79** |
| 3D U-Net                   | 16.21         | 1669.53   | 75.96     | 88.53     | 71.77     | 6.04     | 17.1     | 11.62    |
| Kao et al                  | 9.45          | 203.04    | 78.75     | 90.47     | 81.35     | 3.81     | 4.32     | 7.56     |
| DMFNet                     | 3.86          | 26.93     | 80.12     | 90.62     | 84.54     | 3.06     | 4.66     | 6.31     |
| Partially Reversible U-Net | 3.01          | 956.2     | 80.56     | 90.61     | 85.71     | 3.35     | 5.61     | 7.83     |
| No New-Net                 | 12.43         | 296.83    | 80.66     | 90.92     | 85.22     | 2.74     | 5.83     | 7.2      |
| NVDLMED                    | 40.06         | 1495.53   | 81.73     | 90.68     | 86.02     | 3.82     | 4.41     | 6.84     |

<div  align="center">  
 <img src="https://github.com/easthorse/brain-tumor-segmentation-based-on-group-convolution/blob/base/figure/Figure7.png"
     align=center/>
</div>
<center>Figure 7 The visual comparison of MRI brain tumor segmentation results in horizontal plane, sagittal plane, and coronal plane. The regions in red represent the enhancing tumor and regions in blue represent the necrotic and non-enhancing tumor and the regions in green represent the edema ((a) FLAIR modality of the brain tumor MRI;(b) Segmentation results of classic 3D U-Net;(c) The segmentation results of Our algorithm;(d) The segmentation results manually labeled by experts)</center>

## Conclusion

** **Aiming at the problems of large memory and slow segmentation speed in the
field of computer-aided brain tumor segmentation, an algorithm combining group
convolution and channel shuffle unit is proposed punishment intensity of sparse
class classification error to model is improved  the weighted mixed loss
function to balance the training intensity of different segmentation difficulty
categories. The experimental results show that the algorithm can significantly
reduce the computational cost while maintaining high accuracy provide a powerful
reference for clinicians in brain tumor segmentation.
