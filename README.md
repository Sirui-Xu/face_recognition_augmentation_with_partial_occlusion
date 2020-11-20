# Face Recognition Degradation and Augmentation via Partial Occlusion

The existing supervised algorithms for face recognition have obtained very good results, but in fact, if these algorithms are directly transferred to the face recognition problem with partial occlusion constraints, the performance will be greatly reduced.

Explore the performance degradation by occlusion and regarding augmentation to solve the problem.

### Experiment pipeline

Parameters can be changed in `config.py`

`python train.py`

`python test.py`

### Results

|  Augmentation   |  Baseline  |  w/ Augmentation |
|  ----  | ----  | ---- |
|  no mask  | 0.97733 |  0.98417 |
|  fix mask  | / |  0.97980  |
|  random mask  | / |  0.97333  | 

|  occlusion 1/16   |  Baseline  |  w/ Augmentation |
|  ----  | ----  | ---- |
|  left eye  |  0.95883  |  0.98017  |
|  right eye  |  0.96550  |  0.98267  |
|  nose  |  0.94150  |  0.97717  |
|  left mouth  |  0.97233  |  0.98050  |
|  right mouth  |  0.97000  |  0.98350  |

|  occlusion 1/4   |  Baseline  |  w/ Augmentation |
|  ----  | ----  | ---- |
|  left eye  |  0.91283  |  0.95866  |
|  right eye  |  0.91633  |  0.95883  |
|  nose  |  0.83450  |  0.92633  |
|  left mouth  |  0.94300  |  0.97016  |
|  right mouth  |  0.94050  |  0.97010  |


### Requirements
Please follow the requirements in [face.evoLVe.PyTorch](https://github.com/ZhaoJ9014/face.evoLVe.PyTorch)

For testing, you may need a LFW dataset, while all face was added partial occlusion. 
<https://pan.baidu.com/s/1Ew5JZ266bkg00jB5ICt78g>

For training, you need to download CelebA dataset.


### Acknowledge

part in this implementation refers to some details in [face.evoLVe.PyTorch](https://github.com/ZhaoJ9014/face.evoLVe.PyTorch)
