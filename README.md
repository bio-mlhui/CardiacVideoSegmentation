# Overview
This repo uses the same framework as **[MyFramework](https://github.com/bio-mlhui/MyFramework)**. 

All experiments are based on the **[CardiacUDC](https://www.kaggle.com/datasets/xiaoweixumedicalai/cardiacudc-dataset)** dataset. 

We plan to first run LGRNet (a supervised Non-autoregressive small-model) on the CardiacUDC. After that, we plan to validate several novel designs including "Hybrid Temporal Sampling (HTS)", "Self-supervised Temporal Local-to-Global Self-Distillation (LGSD)",  "AutoRegressive Segmentation beats Unet (AR-SEG)".  

You can click links in each part to check the corresponding implementation.


| Model| FLOPS | #PARAMS | Dice | mIou | $\text{log}^1$ | ckpt | prediction video |
| :----: | :----: | :----: | :----: | :----: | :----: | :----: | :----: |
| BASIC | NAN | NAN | 71.4 | 64.3 | NAN | NAN
| +HTS | NAN | NAN | 71.4 | 64.4 | NAN | NAN
| +LGSD | NAN | NAN | NAN | NAN | NAN | NAN
| BOX-SAM2 | NAN | NAN | NAN | NAN| NAN | NAN
| AR-SEG | NAN | NAN | NAN | NAN | NAN | NAN

###### 1: The log is colored, so you need to use the vscode ANSI extention to view the log file.

## I.Data Curation
### A. Dataset Statistics
CardiacUDC has 4 classes (LF, RF, LV, RV). The videos come from 6 sites, each site uses different ultrasound device. There are a total of 284 annotated videos and 70 un-annotated videos. Each video has an average of 107 frames. Each annotated video has an average of 6 frames annotated.
A total of 

### B. Background Removal
In each video, large amounts of area are trivial background region. These region will cause additional computation. 
Specifically, we use an **[automatic low-level method](https://github.com/bio-mlhui/CardiacVideoSegmentation/blob/cb80b0b07aa4401e9869280248d1c53a75be0973/data_schedule/vis/cardicUDC/transform_dataset_crop_ultrasound.py#L50)** to remove the background and ighlight the central fanshaped region.

<img src="assets/test.png" width="400"> <img src="assets/test_cropped.png" width="200">

### C. Train/Test Split
Since the videos are sparsely annotated, we build a benchmark where each test sample is a video clip of 11 frames and its 6th frame is annotated. We only compute Dice and IoU on the 6th frame.

Each train sample is a video clip of 7 frames. For a video with N annotated frames and M total frames, we can generate N supervised samples and M unsupervised samples. 

## II. BASIC: Cardiac Video Segmentation with LGRNet
We first start with the most common one. The BASIC model has three parts: multi-scale backbone, multi-scale encoder, query-based decoder. 
### To Reproduce:
1. Download the CardiacUDC data to $DATASET_PATH/cardiacUDC

2. move to /data_schedule/vis/cardicUDC
3. change the "root" in transform_dataset_crop_ultrasound to $DATASET_PATH/cardiacUDC
4. python transform_dataset_crop_ultrasound.py
5. download PvT2-B2 pretrained model into $PT_PATH
```
wget -P $PT_PATH/pvt_v2/pvt_v2_b2.pth  https://huggingface.co/huihuixu/lgrnet_ckpts/blob/main/pvt_v2_b2.pth  
```
6. run with 2 24GB GPUS
```
CURRENT_TASK=VIS CUDA_VISIBLE_DEVICES=0,1 WANDB_MODE=online TORCH_NUM_WORKERS=8 python main.py --disable_wandb --trainer_mode train_attmpt --config_file output/VIS/card/supervised/supervised.py
```

### A. Multiscale Backbone
Multi-scale information is vital to Segmentation. We use the Metaformer model, which maps $V\in R^{3\times T \times H\times W}$ to $4\times$, $8\times$, $16\times$, $32\times$ feature maps. 

$\textbf{My understainding to Convolution, Attention, Mamba.}$ Metaformer has 5 stages. The 1st stage is patch embedding, which is the same to ViT. The 2nd and 3rd stages are convolution. The 4th and 5th stages are attention. Why the authors choose this design?

Image information are redundant in surrounding pixels, especially in larger scales. It is not necessary to token-mix all tokens in the image during early stages. (Why? Convolution, Attention, and Mamba are different token-mixers, which provide different views for interacting a set of tokens. If the input tokens each contain different information, then the interaction will be effective. This is why in NLP, the notion of "multi-scale" is not popular. Each word is a sufficiently highly-abstracted meaning, not like low-level pixels. To make token-mixing effective, it is necessay to let each token contain different information. ) (Why Convolution is used to token-mix same-information tokens? Different from Attention and Mamba which are $\textbf{in-context}$ token-mixers, Conlution is context-free. All these token-mixers can be thought as "each output token is a linear combination of input tokens". However, the linear combination weights is fixed for convolution, and dynamic for Attention and Mamba. If the weights are dynamic and predicted from input tokens, the token mixer will be able to in-context learning. You can read more details "Preliminary of Selective State Space Model" in Section 2.2 of my LGRNet paper. I wrote all parts of the LGRNet paper.)

After two downsampling layers, the $16\times$ tokens have aggregate some higher-level information, we can use in-context token-mixers to interact these tokens.

### B. Multiscale Encoder: [h8w8 h16w16 h32w32] -> [h8w8 h16w16 h32w32]
Although the backbone outputs multi-scale features, it seems attractive and reasonable to token-mix tokens from different scales. Deformable Attention is a token-mixer where each input has scale-specific information. Inside **[DeformAttn](https://github.com/bio-mlhui/CardiacVideoSegmentation/blob/cb80b0b07aa4401e9869280248d1c53a75be0973/models/encoder/ops/modules/ms_deform_attn.py#L99)**, we use linear layer to query the position and dynamic weight of each attened token (c -> 1 + 4: weight + position). The output token is the linear combination of features of attened tokens.

### C. FPN: [h4 w4], [h8 w8] -> [h4 w4]
The information has interacted many times for the 8,16,32 scales. However, the 4-scale only has low-level information. To propogate the high-level information to the 4-scale, a simple **[FPN](hhttps://github.com/bio-mlhui/CardiacVideoSegmentation/blob/cb80b0b07aa4401e9869280248d1c53a75be0973/models/encoder/msdeformattn_localGlobal.py#L398)** is used to fuse the 8-scale and 4-scale features.

### D. Mask2former Decoder: [n], [h8w8 h16w16 h32w32] -> [n]
The decoder is just a set of cross-self-ffn layers. The Mask2Former use masked cross attention to improve convergence.
$\textbf{FFN layer.}$ Most model parameters come from the FFN layer, which is composed of two fully-connected layers. The hidden dim is often set to the 4x of the input dimension. A explanation of why doing this (c->4c, 4c->c) is viewing the first transformation as "question answer" and second transformation as "information aggregation". The model asks a number of 4c questions in form of dot-product in the c-space (N c, c 4c -> N 4c), and uses the learned aggregation matrix to output the final representation for each output token (N 4c, 4c c -> N c). This interpretation comes from **[3Blue1Brown](https://www.youtube.com/watch?v=wjZofJX0v4M)**

### E. Mask Head and Class Head: [n c], [c K] -> n K; [n c], [c D], [D H W] -> [n H W]

### F. Bipartie Matching
I remeber it took a lot of time to fully understand the code implementation of "Bipart Matching loss" in the **[DETR](https://github.com/bio-mlhui/CardiacVideoSegmentation/blob/cb80b0b07aa4401e9869280248d1c53a75be0973/models/decoder/mask2former_video3.py#L132)**. In my code, I implement all the bapartite matching loss by myself, instead of using the DETR or Mask2Former implementation.




## III. Hybrid Temporal Sampling during Training
This idea comes from **[HTML](https://openaccess.thecvf.com/content/ICCV2023/papers/Han_HTML_Hybrid_Temporal-scale_Multimodal_Learning_Framework_for_Referring_Video_Object_ICCV_2023_paper.pdf)**. For example we want to sample a traininig clip of 7 frames based on the 15th frame of one video, a naive method is to build the training sample using [12,13,14, 15, 16,17,18]th fraems. However, if the training sample is not continuous such as [1, 5, 13, 15, 20, 40, 100] and we randomly sample frames at different intervals during the whole training process, the model can learn to "link"/"register" same region at different frames. The code is at **[Sample Interval](https://github.com/bio-mlhui/CardiacVideoSegmentation/blob/cb80b0b07aa4401e9869280248d1c53a75be0973/data_schedule/vis/vis_frame_sampler.py#L46)**


## IV. Semi-supervised Cardiac Video Segmentation
Since there are many unlabeled frames in the dataset, I am wondering using the unsupervised signal to update the model. This idea comes from DINO, where each training image is cropped to 2 bigger crops and 9 smaller crops. 11 crops are input to the ViT. And we get 11 [CLS] tokens after the ViT layers. The Self-Distillation loss is to treat the bigger crop as gt probability distribution and smaller crop as predicted distribution, and uses the cross entropy to compute a total of 18 loss values. This loss can train the model to learn the 'local-to-global' correspondance in a self-supervised manner.

## V. Decoder-only AutoRegressive Segmentation beats Unet, Segformer, and Mask2Former for Medical Image Segmentation.

### A. Related Works
There are many works using Unet and Transofmer for Segmentation taks, such as Unet, Mask2fomer. However, these models can not be intergrated into the modern LLM regime with a focus of **[Next-token Prediction is all you need](https://arxiv.org/abs/2409.18869)**. 

Several auto-regressive segmentation 

### B. Proposed Method
Since I am doing LLM, where most researchers adopts the same regime of "Auto-regressive Next-token prediction", I am wondering if the "next-token prediction" can be effective in the Cardiac Video Segmentation. 


### C. Tokenization

### LISA [SEG] token

