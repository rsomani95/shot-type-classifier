<p align="justify"><h1> AI For Filmmaking </h1> </p>

Code for the blog post [AI for Filmmaking](https://rsomani95.github.io/ai-film-1.html). Detect cinematic [shot types](https://rsomani95.github.io/ai-film-1.html#dataset) in images using a pre-trained `ResNet-50`.

This model recognises 6 shot types:

<p align="justify"><h3>1. Extreme Wide Shot</h3></p>

<p align='center'>
  <img src='README_imgs/10cloverfield003-cafe7a6c-3fea-4e44-ad6a-3cf1de4327ab.jpg_cropped.jpg' width='425'/> <img src='README_imgs/chaplin_456-50cfefb3-a1d1-4504-944f-7ca531a9eda6.jpg_cropped.jpg' width='425'/>
</p>

<p align="justify"><h3>2. Long Shot </h3></p>
<p align='center'>
  <img src='README_imgs/gangs-of-new-york_3095-6f49648f-895c-45d5-9d74-3353dab7a662.jpg_cropped.jpg' width='425'/> <img src='README_imgs/google_24-best-wide-shots-images-on-pinterest-wide-angle-shot-e7c680ea-f24a-4ebf-8496-8a51e2705834.jpg_cropped.jpg' width='425'/>
</p>

<p align="justify"><h3>3. Medium Shot </h3></p>

<p align='center'>
  <img src='README_imgs/good-will-hunting_goodwillhunting061-9c782ccc-5914-4793-b7e9-07684474e4d4.jpg_cropped.jpg' width='425'/> <img src='README_imgs/catch-me-if-you-can_3124-ac13629a-5b69-4eb5-8c9b-6e8a25686f0b.jpg_cropped.jpg' width='425'/>
</p>

<p align="justify"><h3>4. Medium Close Up </h3></p>

<p align='center'>
  <img src='README_imgs/10cloverfield023-14265eca-3172-493b-859e-2462f918e8ad.jpg_cropped.jpg' width='425'/> <img src='README_imgs/taxi-driver_53-mohawk-a156a47e-a838-4817-8a65-6e93e4a42473.png_cropped.jpg' width='425'/>
</p>

<p align="justify"><h3>5. Close Up </h3></p>

<p align='center'>
  <img src='README_imgs/fight-club_02-bob-36d930db-f1c3-4bb8-9adc-c7062d3be730.png_cropped.jpg' width='425'/> <img src='README_imgs/google_71751692195015188803-446420bf-5a39-48b6-8570-be45b7d924a3.png_cropped.jpg' width='425'/>
</p>

<p align="justify"><h3>6. Extreme Close Up </h3></p>

<p align='center'>
  <img src='README_imgs/the-fifth-element_6393-793712f1-23f4-43fb-a015-642d34052fe1.jpg_cropped.jpg' width='425'/> <img src='README_imgs/vimeo_fincher_ecu_21-c712f289-aca6-4c74-badc-0b1f4764f151.png_cropped.jpg' width='425'/>
</p>

<br>

In the not so distant future, it will also recognise:

<p align="justify"><h3> Wide Shots </h3></p>

<p align='center'>
  <img src='README_imgs/alice-in-wonderland_aliceinwonderland42-867e6dcb-d014-4a0e-8a8b-78a6d07833da.jpg_cropped.jpg' width='425'/> <img src='README_imgs/2001_space_odyssey_14-hallway1-58e19e02-a14b-46c8-81e0-2f87dd806b66.png_cropped.jpg' width='425'/>
</p>

<p align="justify"><h3> Medium Long Shots </h3></p>

<p align='center'>
  <img src='README_imgs/12_years_a_slave_1733-a1c60a95-f888-4e0f-95bd-76deab52d637.jpg_cropped.jpg' width='425'/> <img src='README_imgs/ex-machina_05-f38cd0f1-e4a2-44df-b833-5c7b0e3376e0.jpg_cropped.jpg' width='425'/>
</p>

<h3> Requirements </h3>

`fastai` — Installation instructions [here](https://docs.fast.ai/install.html). You can use this code _without a GPU._

<h3> Usage </h3>

After downloading the directory, run `bash get_data_model.sh` to download the model and the validation set. A dummy training set is downloaded too to enable the generation of heatmaps.

<h4> Predict Shot Types </h4>

```bash
python get-preds.py                     \
    --path_base  ~/shot-type-classifier \
    --path_img   ~/images               \
    --path_preds ~/images/preds         \
```

Where `path_base` is the directory path, `path_img` the path to the images you want to evaluate, and `path_preds` where you'd like to store the predictions (`.csv` files). The script create the `~/images/preds` folder if it doesn't exist.

<p align='center'>
  <img src='README_imgs/shot13-0001-ccef6be9-7bda-435c-8c75-d2dfcc768e94.png' width='425'/> <img src='README_imgs/preds.png' width='380'/>
</p>

<h4> Heatmaps </h4>

```bash
python get-heatmaps.py                 \
    --path_base ~/shot-type-classifier \
    --path_img  ~/images               \
    --path_hms  ~/images/heatmaps      \
    --alpha 0.8
```

Where `path_base` is the directory path, `path_img` the path to the images you want to evaluate, `path_hms` where you'd like to store the heatmaps, and `alpha` the blending value of the heatmap with the original image. An `alpha` value of `1.0` produces the heatmap only. The script create the `~/images/heatmaps` folder if it doesn't exist.

<p align='center'>
  <img src="README_imgs/tree-of-life1-49fe5c84-1c97-4a03-b245-97062d214db4.jpg" width="425"/> <img src="README_imgs/img_2_heatmap-21d792f5-3826-4e9e-af57-b1b18d9210a2.png" width="411"/>
</p>

<p align="justify"><h3>1. License</h3></p>
This repository is released under the [Creative Commons Attribution-NonCommercial 4.0 (CC BY-NC 4.0)](https://github.com/rsomani95/shot-type-classifier/blob/master/LICENSE). See [here](https://creativecommons.org/licenses/by-nc/4.0/) for more details.
