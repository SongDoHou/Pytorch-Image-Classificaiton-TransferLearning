# Pytorch-Image-Classificaiton-using-Transfer-Learning
 Transfer Learning in Image Classification using pytorch
 
 
Argument 
-d: Directory of your Dataset to training


-b: batch size 


-n: number of nodes in hidden layer 512


-e: number of epochs for training


-t: Directory of your test datset to evaluate your model


Make sure that your dataset is following [this](https://pytorch.org/docs/stable/torchvision/datasets.html#imagefolder) rules


example

python train.py -d ./trainset -b 64 -n 512 -e 20 -t ./test
