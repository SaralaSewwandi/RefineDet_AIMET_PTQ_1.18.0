

given repo
https://github.com/dd604/refinedet.pytorch

SaralaSewwandi



go to (base) bmw@BMW:~/anaconda3/envs$
clone the same virtual enviornment with the previous repo set up

conda create --name refinedetrepo2 --clone refinedet37

conda activate refinedetrepo2 


cd /home/bmw/anaconda3/envs/refinedetrepo2/refinedet.pytorch/

git clone https://github.com/dd604/refinedet.pytorch.git

deleted .git

downloaded vgg16_refinedet320_voc_120000.pth
created a folder called output
uploaded the vgg16_refinedet320_voc_120000.pth to output folder

python -u eval_refinedet.py --input_size 320 --dataset voc --network vgg16 --model_path "./output/vgg16_refinedet320_voc_120000.pth"

ModuleNotFoundError: No module named 'scipy'


conda install -c anaconda scipy


ModuleNotFoundError: No module named 'easydict'
conda install -c conda-forge easydict

ModuleNotFoundError: No module named 'pycocotools'

conda install -c conda-forge pycocotools

create  data folder
"/home/bmw/anaconda3/envs/refinedetrepo2/refinedet.pytorch/data/"

cd data/
cp -r  ~/data/VOCdevkit/ .


python -u eval_refinedet.py --input_size 320 --dataset voc --network vgg16 --model_path "./output/vgg16_refinedet320_voc_120000.pth"


AssertionError: Path does not exist: /home/bmw/anaconda3/envs/refinedetrepo2/refinedet.pytorch/data/VOCdevkit2007/VOC2007/ImageSets/Main/test.txt


"/home/bmw/anaconda3/envs/refinedetrepo2/refinedet.pytorch/data/VOCdevkit/VOC2007/ImageSets/Main/test.txt"
/home/bmw/anaconda3/envs/refinedetrepo2/refinedet.pytorch/data/VOCdevkit2007/VOC2007/ImageSets/Main/test.txt

cp -r "/home/bmw/anaconda3/envs/refinedetrepo2/refinedet.pytorch/data/VOCdevkit/VOC2007/" "/home/bmw/anaconda3/envs/refinedetrepo2/refinedet.pytorch/data/VOCdevkit2007/"


python -u eval_refinedet.py --input_size 320 --dataset voc --network vgg16 --model_path "./output/vgg16_refinedet320_voc_120000.pth"
OK

cd /home/bmw/anaconda3/envs/refinedetrepo2/refinedet.pytorch/
git clone https://github.com/SaralaSewwandi/refinedet-pytorch.git

(refinedetrepo2) bmw@BMW:~/anaconda3/envs/refinedetrepo2/refinedet.pytorch$ cp -r refinedet-pytorch/.git .

(refinedetrepo2) bmw@BMW:~/anaconda3/envs/refinedetrepo2/refinedet.pytorch$ rm -r refinedet-pytorch/



=============
downloded
vgg16_refinedet512_voc_120000.pth
upload to output folder


python -u eval_refinedet.py --input_size 512 --dataset voc --network vgg16 --model_path "./output/vgg16_refinedet512_voc_120000.pth"

=======map value calculation=============

mAp value is printed in "/home/bmw/anaconda3/envs/refinedetrepo2/refinedet.pytorch/libs/datasets/pascal_voc.py" file

"/home/bmw/anaconda3/envs/refinedetrepo2/refinedet.pytorch/libs/datasets/voc_eval.py" is also used

==========================


RefineDet(Repo)	VGG16	320 x 320	78.6   RefineDet(Our)	VGG16	320 x 320	Mean AP = 0.7853
RefineDet(Repo)	VGG16	512 x 512	79.1   RefineDet(Our)	VGG16	512 x 512	Mean AP = 0.7914

"/home/bmw/anaconda3/envs/refinedetrepo2/refinedet.pytorch/refinedet320_eval_results.txt"
"/home/bmw/anaconda3/envs/refinedetrepo2/refinedet.pytorch/refinedet512_eval_results.txt"


(refinedetrepo2) bmw@BMW:~/anaconda3/envs/refinedetrepo2/refinedet.pytorch$ cd demo/
(refinedetrepo2) bmw@BMW:~/anaconda3/envs/refinedetrepo2/refinedet.pytorch/demo$ python demo_refinedet_320.py
"/home/bmw/anaconda3/envs/refinedetrepo2/refinedet.pytorch/demo/000004refinedet320_result.jpg" generated

(refinedetrepo2) bmw@BMW:~/anaconda3/envs/refinedetrepo2/refinedet.pytorch$ cd demo/
(refinedetrepo2) bmw@BMW:~/anaconda3/envs/refinedetrepo2/refinedet.pytorch/demo$ python demo_refinedet_512.py
"/home/bmw/anaconda3/envs/refinedetrepo2/refinedet.pytorch/demo/000004refinedet512_result.jpg" generated







