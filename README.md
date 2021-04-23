# Image Captioning Inference

This project use models and weights from [self-critical.pytorch](https://github.com/ruotianluo/self-critical.pytorch).
Bottom-up attention embeddings generated using [py-bottom-up-attention](https://github.com/airsplay/py-bottom-up-attention), which is pytorch implementation of [bottom-up-attention](https://github.com/peteanderson80/bottom-up-attention).

![](/readme_pics/example.png)

## Requirements
- requirements.txt

## Install

You should install detectron with
```
python3 setup.py build develop
```

Also you should download weights for ResNet, Bottom-up attention models.

Or you can install and download models using `download.sh` script.

Repo installation code, which I use:
```
git clone https://github.com/grazder/Image-Captioning-Inference.git
cd Image-Captioning-Inference
pip install -r requirements.txt
bash download.sh
```

You don't need to download all models only models which you will use.
For example: bottom-up attention + transformer. Everything else you can comment.

## Models

There are a lot of models from [self-critical.pytorch](https://github.com/ruotianluo/self-critical.pytorch). Which you can find in [MODEL_ZOO](https://github.com/ruotianluo/self-critical.pytorch/blob/master/MODEL_ZOO.md).

## Object initialization and usage example
```
from Captions import Captions
import os

model_fc_resnet = Captions(
                  model_path='data/fc-resnet-weights/model.pth',
                  infos_path='data/fc-resnet-weights/infos.pkl',
                  model_type='resnet',
                  resnet_model_path='data/imagenet_weights/resnet101.pth',
                  bottom_up_model_path='data/bottom-up/faster_rcnn_from_caffe.pkl',
                  bottom_up_config_path='data/bottom-up/faster_rcnn_R_101_C4_caffe.yaml',
                  bottom_up_vocab='data/vocab/objects_vocab.txt',
                  device='cpu'
                  )

images = os.listdir('example_images/')
paths = [os.path.join('example_images', x) for x in images]

preds = model_fc_resnet.get_prediction(paths)

for i, pred in enumerate(preds):
    print(f'{images[i]}: {pred}')
```

## Models Timings:
I took scores and models from [MODEL_ZOO](https://github.com/ruotianluo/self-critical.pytorch/blob/master/MODEL_ZOO.md).
Time estimated in **google colab**.

### Trained with Resnet101 feature:

Collection: [link](https://drive.google.com/open?id=0B7fNdx_jAqhtcXp0aFlWSnJmb0k)

<table><tbody>
<!-- START TABLE -->
<!-- TABLE HEADER -->
<th valign="bottom">Name</th>
<th valign="bottom">CIDEr</th>
<th valign="bottom">SPICE</th>
<th valign="bottom">Download</th>
<th valign="bottom">Time @ 7 images.</th>
<!-- TABLE BODY -->
 <tr><td align="left"><a href="configs/fc.yml">FC</a></td>
<td align="center">0.953</td>
<td align="center">0.1787</td>
<td align="center"><a href="https://drive.google.com/open?id=1AG8Tulna7gan6OgmYul0QhxONDBGcdun">model&metrics</a></td>
<td align="center">28.4 s</td>
</tr>
 <tr><td align="left"><a href="configs/fc_rl.yml">FC<br>+self_critical</a></td>
<td align="center">1.045</td>
<td align="center">0.1838</td>
<td align="center"><a href="https://drive.google.com/open?id=1MA-9ByDNPXis2jKG0K0Z-cF_yZz7znBc">model&metrics</a></td>
<td align="center">28.6 s</td>
</tr>
 <tr><td align="left"><a href="configs/fc_nsc.yml">FC<br>+new_self_critical</a></td>
<td align="center">1.053</td>
<td align="center">0.1857</td>
<td align="center"><a href="https://drive.google.com/open?id=1OsB_jLDorJnzKz6xsOfk1n493P3hwOP0">model&metrics</a></td>
<td align="center">29.9 s</td>
</tr>
</tbody></table>

### Trained with Bottomup feature (10-100 features per image, not 36 features per image):

Collection: [link](https://drive.google.com/open?id=1-RNak8qLUR5LqfItY6OenbRl8sdwODng)

<table><tbody>
<!-- START TABLE -->
<!-- TABLE HEADER -->
<th valign="bottom">Name</th>
<th valign="bottom">CIDEr</th>
<th valign="bottom">SPICE</th>
<th valign="bottom">Download</th>
<th valign="bottom">Time @ 7 images.</th>
<!-- TABLE BODY -->
 <tr><td align="left"><a href="configs/a2i2.yml">Att2in</a></td>
<td align="center">1.089</td>
<td align="center">0.1982</td>
<td align="center"><a href="https://drive.google.com/open?id=1jO9bSocC93n1vBZmZVaASWc_jJ1VKZUq">model&metrics</a></td>
<td align="center">2min 17s</td>
</tr>
 <tr><td align="left"><a href="configs/a2i2_sc.yml">Att2in<br>+self_critical</a></td>
<td align="center">1.173</td>
<td align="center">0.2046</td>
<td align="center"><a href="https://drive.google.com/open?id=1aI7hYUmgRLksI1wvN9-895GMHz4yStHz">model&metrics</a></td>
<td align="center">2min 18s</td>
</tr>
 <tr><td align="left"><a href="configs/a2i2_nsc.yml">Att2in<br>+new_self_critical</a></td>
<td align="center">1.195</td>
<td align="center">0.2066</td>
<td align="center"><a href="https://drive.google.com/open?id=1BkxLPL4SuQ_qFa-4fN96u23iTFWw-iXX">model&metrics</a></td>
<td align="center">2min 19s</td>
</tr>
 <tr><td align="left"><a href="configs/updown/updown.yml">UpDown</a></td>
<td align="center">1.099</td>
<td align="center">0.1999</td>
<td align="center"><a href="https://drive.google.com/open?id=14w8YXrjxSAi5D4Adx8jgfg4geQ8XS8wH">model&metrics</a></td>
<td align="center">2min 20s</td>
</tr>
 <tr><td align="left"><a href="configs/updown/updown_sc.yml">UpDown<br>+self_critical</a></td>
<td align="center">1.227</td>
<td align="center">0.2145</td>
<td align="center"><a href="https://drive.google.com/open?id=1QdCigVWdDKTbUe3_HQFEGkAsv9XIkKkE">model&metrics</a></td>
<td align="center">2min 19s</td>
</tr>
 <tr><td align="left"><a href="configs/updown/updown_nsc.yml">UpDown<br>+new_self_critical</a></td>
<td align="center">1.239</td>
<td align="center">0.2154</td>
<td align="center"><a href="https://drive.google.com/open?id=1cgoywxAdzHtIF2C6zNnIA7G2wjol_ybf">model&metrics</a></td>
<td align="center">2min 19s</td>
</tr>
 <tr><td align="left"><a href="configs/updown/ud_long_nsc.yml">UpDown<br>+Schedule long<br>+new_self_critical</a></td>
<td align="center">1.280</td>
<td align="center">0.2200</td>
<td align="center"><a href="https://drive.google.com/open?id=1bCDmf4JCM79f5Lqp6MAn1ap4b3NJ5Gis">model&metrics</a></td>
<td align="center">2min 20s</td>
</tr>
 <tr><td align="left"><a href="configs/transformer/transformer.yml">Transformer</a></td>
<td align="center">1.1259</td>
<td align="center">0.2063</td>
<td align="center"><a href="https://drive.google.com/open?id=10Q5GJ2jZFCexD71rY9gg886Aasuaup8O">model&metrics</a></td>
<td align="center">2min 21s</td>
</tr>
<tr><td align="left"><a href="configs/transformer/transformer_step.yml">Transformer(warmup+step decay)</a></td>
<td align="center">1.1496</td>
<td align="center">0.2093</td>
<td align="center"><a href="https://drive.google.com/drive/folders/1Qog9yvpGWdHanFXFITjyrWXMzre3ek3e?usp=sharing">model&metrics</a></td>
<td align="center">Although this schedule is better, the final self critical results are similar.</td>
</tr>
 <tr><td align="left"><a href="configs/transformer/transformer_scl.yml">Transformer<br>+self_critical</a></td>
<td align="center">1.277</td>
<td align="center">0.2249</td>
<td align="center"><a href="https://drive.google.com/open?id=12iKJJSIGrzFth_dJXqcXy-_IjAU0I3DC">model&metrics</a></td>
<td align="center">This could be higher in my opinion. I chose the checkpoint with the highest CIDEr on val set, so it's possible some other checkpoint may perform better. Just let you know.</td>
</tr>
 <tr><td align="left"><a href="configs/transformer/transformer_nscl.yml">Transformer<br>+new_self_critical</a></td>
<td align="center"><b>1.303</b></td>
<td align="center">0.2289</td>
<td align="center"><a href="https://drive.google.com/open?id=1sJDqetTVOnei6Prgvl_4vkvrYlKlc-ka">model&metrics</a></td>
<td align="center"></td>
</tr>
</tbody></table>

## Captions Examples

<table><tbody>
<!-- START TABLE -->
<!-- TABLE HEADER -->
<th valign="bottom">Name</th>
<th valign="bottom">street.jpg</th>
<th valign="bottom">man.jpeg</th>
<th valign="bottom">statue.jpeg</th>
<th valign="bottom">tv_man.jpeg</th>
<!-- TABLE BODY -->
<tr><td align="left">**FC**</td>
<td align="center">a group of people walking down a street</td>
<td align="center">a man in a suit and tie holding a cell phone</td>
<td align="center">a man in a hat and a hat holding a frisbee</td>
<td align="center">a man is brushing his teeth with a tooth brush</td>
</tr>

<tr><td align="left">**FC + self-critical**</td>
<td align="center">a group of people riding a bike down a street</td>
<td align="center">a man wearing a suit and a tie</td>
<td align="center">a man standing next to a man with a baseball bat</td>
<td align="center">a man taking a picture in a bathroom with a mirror</td>
</tr>

<tr><td align="left">**FC + new-self-critical**</td>
<td align="center">a group of people riding bikes down a city street</td>
<td align="center">a man wearing a suit and tie talking on a cell phone</td>
<td align="center">a man is holding a frisbee in a street</td>
<td align="center">a man brushing his teeth in a bathroom with a mirror</td>
</tr>

<tr><td align="left">**Att2in**</td>
<td align="center">a group of people riding bikes down a city street</td>
<td align="center"> a man in a suit and tie is wearing a suit</td>
<td align="center">a man and a woman are standing in a park</td>
<td align="center">a man in a blue shirt playing a video game</td>
</tr>

<tr><td align="left">**Att2in + self-critical**</td>
<td align="center">a group of people riding a bike down a city street</td>
<td align="center">a man wearing a suit and tie and a table</td>
<td align="center">a man and a woman sitting on a bench with a book</td>
<td align="center">a man playing a video game in a wii</td>
</tr>

<tr><td align="left">**Att2in + new self-critical**</td>
<td align="center">a group of people riding bikes down a city street</td>
<td align="center">a man in a suit and tie standing in front of a table</td>
<td align="center">a man and a woman sitting on a bench with a book</td>
<td align="center">a man is playing a video game with a wii</td>
</tr>

<tr><td align="left">**Updown**</td>
<td align="center">a group of people riding bikes down a street</td>
<td align="center">a man in a suit and tie is holding a microphone</td>
<td align="center">a man and a woman are standing in front of a tree</td>
<td align="center">a man is playing a video game on a television</td>
</tr>

<tr><td align="left">**Updown + self-critical**</td>
<td align="center">a group of people riding bikes down a city street</td>
<td align="center">a man in a suit and tie sitting on a table</td>
<td align="center">a man and a woman sitting on a bench with a book</td>
<td align="center">a man is holding a video game on a television</td>
</tr>

<tr><td align="left">**Updown + new self-critical**</td>
<td align="center">a group of people riding bikes down a city street</td>
<td align="center">a man in a suit and tie in a UNK</td>
<td align="center">a man and a woman holding a book</td>
<td align="center">a man is playing a video game on a tv</td>
</tr>

<tr><td align="left">**UpDown+Schedule long+new_self_critical**</td>
<td align="center">a group of people riding on a city street</td>
<td align="center">a man in a suit and tie sitting in a table</td>
<td align="center">a man and a woman standing in front of a tree</td>
<td align="center">a man playing a video game with a wii</td>
</tr>

<tr><td align="left">**Transformer**</td>
<td align="center">a group of people are riding bikes on the sidewalk</td>
<td align="center">a man in a suit and tie sitting in a chair</td>
<td align="center">a man and woman standing in front of a statue</td>
<td align="center">a man in a room playing a video game</td>
</tr>

<tr><td align="left">**Transformer(warmup+step decay)**</td>
<td align="center">a group of people riding bikes down a city street</td>
<td align="center">a man in a suit sitting in a chair</td>
<td align="center">a man and woman standing next to each other</td>
<td align="center">a man is playing a video game on a large screen</td>
</tr>

<tr><td align="left">**Transformer + self-critical**</td>
<td align="center">a group of people riding bikes down a city street</td>
<td align="center">a man in a suit and tie sitting in a room</td>
<td align="center">a man and a woman standing in front of a tree</td>
<td align="center">a man playing a video game in a room</td>
</tr>

<tr><td align="left">**Transformer + new self-critical**</td>
<td align="center">a group of people riding bikes down a city street</td>
<td align="center">a man in a suit and tie sitting in a room</td>
<td align="center">a man and a woman standing next to a tree</td>
<td align="center">a man sitting in front of a television</td>
</tr>

</tbody></table>
