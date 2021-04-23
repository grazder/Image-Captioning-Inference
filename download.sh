#!/bin/bash

python3 setup.py build develop

### Embeddings
# ResNet Weights
gdown --id 0B7fNdx_jAqhtSmdCNDVOVVdINWs -O resnet101.pth
mkdir data/imagenet_weights/
mv resnet101.pth data/imagenet_weights/

# Bottom-up Attention Features Weights
wget http://nlp.cs.unc.edu/models/faster_rcnn_from_caffe.pkl
mkdir data/bottom-up/
mv faster_rcnn_from_caffe.pkl data/bottom-up/

### Models
# FC Weights
# https://drive.google.com/drive/folders/1AG8Tulna7gan6OgmYul0QhxONDBGcdun
gdown --id 1AFUo86ncktEceQZVhzPEMFc0srnZBNGk -O infos.pkl
mkdir data/fc-weights/
mv infos.pkl data/fc-weights/

gdown --id 1eBIV4JzVT1DYQveb3j_FBxuDv1XnqdA7 -O model.pth
mkdir data/fc-weights/
mv model.pth data/fc-weights/

# FC + self_critical
# https://drive.google.com/drive/folders/1MA-9ByDNPXis2jKG0K0Z-cF_yZz7znBc
gdown --id 1GJUzpw0IyHaw7-zTYoXKjBe28qcxz7OR -O infos.pkl
mkdir data/fc-self-critical-weights/
mv infos.pkl data/fc-self-critical-weights/

gdown --id 1wRQ9XZjjZymN0J07P9UMqLFlFMBIxO3O -O model.pth
mkdir data/fc-self-critical-weights/
mv model.pth data/fc-self-critical-weights/

# FC + new_self_critical Weights
# https://drive.google.com/drive/folders/1OsB_jLDorJnzKz6xsOfk1n493P3hwOP0
gdown --id 1QG4criqiVuCfur6juvH1ddi2twUS3bJ9 -O infos.pkl
mkdir data/fc-new-self-critical-weights/
mv infos.pkl data/fc-new-self-critical-weights/

gdown --id 1pSfTXtevxSh84LTsV-Ed-nolxPQBxFYQ -O model.pth
mkdir data/fc-new-self-critical-weights/
mv model.pth data/fc-new-self-critical-weights/

# Att2in Weights
# https://drive.google.com/drive/folders/1jO9bSocC93n1vBZmZVaASWc_jJ1VKZUq
gdown --id 1DiMDelXtxoTDMShBhIKv3qKYSZVRgqJV -O model.pth
mkdir data/att2in-weights/
mv model.pth data/att2in-weights/

gdown --id 1sqeE3qgu4DA9fRMKQVjs1g-v5LUC2r5B -O infos.pkl
mkdir data/att2in-weights/
mv infos.pkl data/att2in-weights/

# Att2in + self_critical Weights
# https://drive.google.com/open?id=1aI7hYUmgRLksI1wvN9-895GMHz4yStHz

gdown --id 1NCUSWz1gz9p6VMDT-KTV9Xog3wT6TV5t -O model.pth
mkdir data/att2in-self-critical-weights/
mv model.pth data/att2in-self-critical-weights/

gdown --id 1iTQ3x97ezpDcY2Y-KHh58JPlq2LktFeC -O infos.pkl
mkdir data/att2in-self-critical-weights/
mv infos.pkl data/att2in-self-critical-weights/

# Att2in + new_self_critical Weights
# https://drive.google.com/drive/folders/1BkxLPL4SuQ_qFa-4fN96u23iTFWw-iXX

gdown --id 108FPUUBf429SnophSEzZjYrk9HmRzLc9 -O model.pth
mkdir data/att2in-new-self-critical-weights/
mv model.pth data/att2in-new-self-critical-weights/

gdown --id 1CAReT5XttQ-ha6YDYINGBbBl46z-qE3u -O infos.pkl
mkdir data/att2in-new-self-critical-weights/
mv infos.pkl data/att2in-new-self-critical-weights/

# UpDown Weights
# https://drive.google.com/drive/folders/14w8YXrjxSAi5D4Adx8jgfg4geQ8XS8wH

gdown --id 1YIzkBMEMVlpSKegZ1cF5VCI5LrrpoiMC -O model.pth
mkdir data/updown-weights/
mv model.pth data/updown-weights/

gdown --id 1BBZ15YRJk0y0MVnZukHRMvJ1pFXT-HVh -O infos.pkl
mkdir data/updown-weights/
mv infos.pkl data/updown-weights/

# UpDown + self_critical Weights
# https://drive.google.com/drive/folders/1QdCigVWdDKTbUe3_HQFEGkAsv9XIkKkE

gdown --id 1whwfmukxambp601jQA01dpTJRT_Jg4w2 -O model.pth
mkdir data/updown-self-critical-weights/
mv model.pth data/updown-self-critical-weights/

gdown --id 1lo6rtSdwb_stQy1qxTsvKUQkgdKojcAH -O infos.pkl
mkdir data/updown-self-critical-weights/
mv infos.pkl data/updown-self-critical-weights/

# UpDown + new_self_critical Weights
# https://drive.google.com/drive/folders/1cgoywxAdzHtIF2C6zNnIA7G2wjol_ybf

gdown --id 1wddHVpQ6ygYyIomPFkBQLdc5I74dWnCU -O model.pth
mkdir data/updown-new-self-critical-weights/
mv model.pth data/updown-new-self-critical-weights/

gdown --id 18nunlqwuVUwv1uJeoiBdTVFmeJWBd6zC -O infos.pkl
mkdir data/updown-new-self-critical-weights/
mv infos.pkl data/updown-new-self-critical-weights/

# UpDown + Schedule long + new_self_critical Weights
# https://drive.google.com/drive/folders/1bCDmf4JCM79f5Lqp6MAn1ap4b3NJ5Gis

gdown --id 1dce7Wf6nuTR5CFuFO8xhPDdW4ahaTzwe -O model.pth
mkdir data/updown-yangxuntu-weights/
mv model.pth data/updown-yangxuntu-weights/

gdown --id 1DU-tiArdC3-fWBC-TVCuBmodf9gryor4 -O infos.pkl
mkdir data/updown-yangxuntu-weights/
mv infos.pkl data/updown-yangxuntu-weights/

# Transformer Weights
# https://drive.google.com/drive/folders/10Q5GJ2jZFCexD71rY9gg886Aasuaup8O

gdown --id 1xyswU8iXYkenzxZMthcjNo0nvP4hhV2g -O model.pth
mkdir data/transformer-weights/
mv model.pth data/transformer-weights/

gdown --id 1lD2Em5A5gv7TzoHjS1S1axE53C1yGOV6 -O infos.pkl
mkdir data/transformer-weights/
mv infos.pkl data/transformer-weights/

# Transformer + self_critical Weights
# https://drive.google.com/drive/folders/12iKJJSIGrzFth_dJXqcXy-_IjAU0I3DC

gdown --id 1bztJgN9H4EHuZ6dGtD2D23qPoC0IoZWP -O model.pth
mkdir data/transformer-step-weights/
mv model.pth data/transformer-step-weights/

gdown --id 139y1kaPjCj7EjySBof6w64_M-54af0LS -O infos.pkl
mkdir data/transformer-step-weights/
mv infos.pkl data/transformer-step-weights/

# Transformer + new_self_critical Weights
# https://drive.google.com/drive/folders/1sJDqetTVOnei6Prgvl_4vkvrYlKlc-ka?usp=drive_open

gdown --id 1tt6kaAQW6ZM0i7YcSJuIiAtZ2fbuU3H1 -O model.pth
mkdir data/transformer-new-self-critical-weights/
mv model.pth data/transformer-new-self-critical-weights/

gdown --id 1gvseH4bwghWzPrWfPMwumwbfvJvWYm9k -O infos.pkl
mkdir data/transformer-new-self-critical-weights/
mv infos.pkl data/transformer-new-self-critical-weights/