<h2>TensorFlow-FlexUNet-Tiled-Image-Segmentation-Curated-Spacecrafts (2026/01/22)</h2>
Toshiyuki Arai<br>
Software Laboratory antillia.com<br><br>
This is the first experiment of Image Segmentation for Spacecrafts 
based on our <a href="./src/TensorFlowFlexUNet.py">TensorFlowFlexUNet</a> (TensorFlow Flexible UNet Image Segmentation Model for Multiclass) , 
and a 512x512  pixels <a href="https://drive.google.com/file/d/1w6Wn16IriFcpQLrT44W-6ZoO_mdCJ2sd/view?usp=sharing">
<b>Tiled-Curated-Spacecrafts-ImageMask-Dataset.zip</b></a>, which was derived by us from
 <a href="https://www.kaggle.com/datasets/dkudryavtsev/spacecrafts">
<b>Spacecrafts</b> 
</a> on the kaggle web site.
<br><br>
<hr>
<b>Actual Image Segmentation for Curated-Spacecrafts Images of 1024x512 pixels</b><br>
As shown below, the inferred masks look similar to the ground truth masks, but they lack precision in certain areas,
especially this model failed to detect blue regions. <br>
<br>
<table border=1 style='border-collapse:collapse;' cellpadding='5'>
<tr>
<th>Input: image</th>
<th>Mask (ground_truth)</th>
<th>Prediction: tiled_inferred_mask</th>
</tr>
<tr>
<td><img src="./projects/TensorFlowFlexUNet/Curated-Spacecrafts/mini_test/images/1001.png" width="320" height="auto"></td>
<td><img src="./projects/TensorFlowFlexUNet/Curated-Spacecrafts/mini_test/masks/1001.png" width="320" height="auto"></td>
<td><img src="./projects/TensorFlowFlexUNet/Curated-Spacecrafts/mini_test_output_tiled/1001.png" width="320" height="auto"></td>
</tr>

<tr>
<td><img src="./projects/TensorFlowFlexUNet/Curated-Spacecrafts/mini_test/images/1010.png" width="320" height="auto"></td>
<td><img src="./projects/TensorFlowFlexUNet/Curated-Spacecrafts/mini_test/masks/1010.png" width="320" height="auto"></td>
<td><img src="./projects/TensorFlowFlexUNet/Curated-Spacecrafts/mini_test_output_tiled/1010.png" width="320" height="auto"></td>
</tr>

<tr>
<td><img src="./projects/TensorFlowFlexUNet/Curated-Spacecrafts/mini_test/images/1016.png" width="320" height="auto"></td>
<td><img src="./projects/TensorFlowFlexUNet/Curated-Spacecrafts/mini_test/masks/1016.png" width="320" height="auto"></td>
<td><img src="./projects/TensorFlowFlexUNet/Curated-Spacecrafts/mini_test_output_tiled/1016.png" width="320" height="auto"></td>
</tr>

</table>
<hr>
<br>
<h3>1  Dataset Citation</h3>
The dataset used here was derived from:<br><br>
 <a href="https://www.kaggle.com/datasets/dkudryavtsev/spacecrafts">
<b>Spacecrafts</b>
</a>.
<br>
<b>Spacecraft Dataset for Detection, Segmentation, and Parts Recognition</b>
<br><br>
<b>About Dataset</b><br>
A dataset for object detection and semantic segmentation using both synthetic and real spacecraft images.<br><br>
Downloaded from <a href="https://github.com/Yurushia1998/SatelliteDataset">https://github.com/Yurushia1998/SatelliteDataset</a><br>
Authors : Dung Anh Hoang, Bo Chen, Tat-Jun Chin<br>
See also the paper: <a href="https://arxiv.org/abs/2106.08186">https://arxiv.org/abs/2106.08186</a><br>
<br>
The dataset includes 3116 images and masks with a size of 1280x720 px as well as bounding boxes for both synthetic and real spacecraft images. <br>
Each spacecraft is segmented into at most 3 parts, including the body, solar panel, and antenna, marked by an RGB mask of, respectively, 
3 colors: green, red, and blue.<br>
<br>
The images with indices 0-1002 have fine mask, while images 1003-3116 are with coarse masks. The datasets is divided into 2 parts: 
the train data with 403 fine masks (indices 0-402) and 2114 coarse masks (indices 1003-3116); the validation dataset including 600 images with fine masks indexed from 403 to 1002.
<br>
The file all_bbox.txt contains the bounding boxes for all the spacecrafts inside the datasets in a form of a dictionary with indices of images as the keys. The bounding box format is [max_x, max_y, min_x, min_y].
<br><br>
Warning! Some images may be corrupted, and also there are no masks for a number of the images. <br>
An example of semantic segmentation: <a href="https://www.kaggle.com/code/dkudryavtsev/spacecraft-component-segmentation">
https://www.kaggle.com/code/dkudryavtsev/spacecraft-component-segmentation
</a>
<br><br>
<b>License</b><br>
Unknown
<br>
<br>
<h3>
2 Curated-Spacecrafts ImageMask Dataset
</h3>
 If you would like to train this Spacecrafts Segmentation model by yourself,
 please download the original dataset from the google drive  
<a href="https://drive.google.com/file/d/1w6Wn16IriFcpQLrT44W-6ZoO_mdCJ2sd/view?usp=sharing">
<b>Tiled-Curated-Spacecrafts-ImageMask-Dataset.zip</b></a>
, expand the downloaded, and put it in <b>./dataset </b> folder to be:<br>
<pre>
./dataset
└─Curated-Spacecrafts
    ├─test
    │   ├─images
    │   └─masks
    ├─train
    │   ├─images
    │   └─masks
    └─valid
        ├─images
        └─masks
</pre>
<br>
<b>Curated-Spacecrafts Statistics</b><br>
<img src ="./projects/TensorFlowFlexUNet/Curated-Spacecrafts/Curated-Spacecrafts_Statistics.png" width="512" height="auto"><br>
<br>
As shown above, the number of images of train and valid datasets is not enough to use for a training set of our segmentation model.
<br>
<br>
<b>Train_images_sample</b><br>
<img src="./projects/TensorFlowFlexUNet/Curated-Spacecrafts/asset/train_images_sample.png" width="1024" height="auto">
<br>
<b>Train_masks_sample</b><br>
<img src="./projects/TensorFlowFlexUNet/Curated-Spacecrafts/asset/train_masks_sample.png" width="1024" height="auto">
<br>
<br>
<h3>
3 Train TensorflowFlexUNet Model
</h3>
 We trained Curated-Spacecrafts TensorflowFlexUNet Model by using the following
<a href="./projects/TensorFlowFlexUNet/Curated-Spacecrafts/train_eval_infer.config"> <b>train_eval_infer.config</b></a> file. <br>
Please move to ./projects/TensorFlowFlexUNet/Curated-Spacecrafts and run the following bat file.<br>
<pre>
>1.train.bat
</pre>
, which simply runs the following command.<br>
<pre>
>python ../../../src/TensorFlowFlexUNetTrainer.py ./train_eval_infer.config
</pre>
<hr>

<b>Model parameters</b><br>
Defined a small <b>base_filters=16</b> and a large <b>base_kernels=(11,11)</b> for the first Conv Layer of Encoder Block of 
<a href="./src/TensorflowUNet.py">TensorflowUNet.py</a> 
and a large num_layers (including a bridge between Encoder and Decoder Blocks).
<pre>
[model]
image_width    = 512
image_height   = 512
image_channels = 3
input_normalize = True
normalization  = False
num_classes    = 4
base_filters   = 16
base_kernels   = (11,11)
num_layers     = 8
dropout_rate   = 0.05
dilation       = (1,1)
</pre>

<b>Learning rate</b><br>
Defined a small learning rate.  
<pre>
[model]
learning_rate  = 0.00007
</pre>

<b>Loss and metrics functions</b><br>
Specified "categorical_crossentropy" and "dice_coef_multiclass".<br>
<pre>
[model]
loss           = "categorical_crossentropy"
metrics        = ["dice_coef_multiclass"]
</pre>
<b >Learning rate reducer callback</b><br>
Enabled learing_rate_reducer callback, and a small reducer_patience.
<pre> 
[train]
learning_rate_reducer = True
reducer_factor     = 0.5
reducer_patience   = 4
</pre>
<b>Early stopping callback</b><br>
Enabled early stopping callback with patience parameter.
<pre>
[train]
patience      = 10
</pre>
<b></b><br>
<b>RGB color map</b><br>
rgb color map dict for Curated-Spacecrafts 1+1 classes.<br>
<pre>
[mask]
mask_file_format = ".png"
;Curated-Spacecrafts 1+3
;                       
rgb_map={(0,0,0):0,(255,0,0):1, (0,255,0):2, (0,0,255):3}
</pre>
<b>Epoch change inference callbacks</b><br>
Enabled epoch_change_tild_infer callback.<br>
<pre>
[train]
epoch_change_infer     = False
epoch_change_infer_dir =  "./epoch_change_infer"
epoch_change_tiled_infer     = True
epoch_change_tiled_infer_dir =  "./epoch_change_tiled_infer"
</pre>
By using this epoch_change_tiled_infer callback, on every epoch_change, the inference procedure can be called
 for 6 images in <b>mini_test</b> folder. This will help you confirm how the predicted mask changes 
 at each epoch during your training process.<br> <br> 
<b>Epoch_change_inference output at starting (1,2,3,4)</b><br>
<img src="./projects/TensorFlowFlexUNet/Curated-Spacecrafts/asset/epoch_change_infer_at_start.png" width="1024" height="auto"><br>
<br>
<b>Epoch_change_inference output at middle point (15,16,17,18)</b><br>
<img src="./projects/TensorFlowFlexUNet/Curated-Spacecrafts/asset/epoch_change_infer_at_middlepoint.png" width="1024" height="auto"><br>
<br>
<b>Epoch_change_inference output at ending (32,33,34,35)</b><br>
<img src="./projects/TensorFlowFlexUNet/Curated-Spacecrafts/asset/epoch_change_infer_at_end.png" width="1024" height="auto"><br>

<br>
In this experiment, the training process was stopped at epoch 27 by EarlyStoppingCallback.<br><br>
<img src="./projects/TensorFlowFlexUNet/Curated-Spacecrafts/asset/train_console_output_at_epoch35.png" width="880" height="auto"><br>
<br>
<a href="./projects/TensorFlowFlexUNet/Curated-Spacecrafts/eval/train_metrics.csv">train_metrics.csv</a><br>
<img src="./projects/TensorFlowFlexUNet/Curated-Spacecrafts/eval/train_metrics.png" width="520" height="auto"><br>

<br>
<a href="./projects/TensorFlowFlexUNet/Curated-Spacecrafts/eval/train_losses.csv">train_losses.csv</a><br>
<img src="./projects/TensorFlowFlexUNet/Curated-Spacecrafts/eval/train_losses.png" width="520" height="auto"><br>
<br>

<h3>
4 Evaluation
</h3>
Please move to a <b>./projects/TensorFlowFlexUNet/Curated-Spacecrafts</b> folder,<br>
and run the following bat file to evaluate TensorflowFlexUNet model for Curated-Spacecrafts.<br>
<pre>
>./2.evaluate.bat
</pre>
This bat file simply runs the following command.
<pre>
>python ../../../src/TensorFlowFlexUNetEvaluator.py  ./train_eval_infer.config
</pre>
Evaluation console output:<br>
<img src="./projects/TensorFlowFlexUNet/Curated-Spacecrafts/asset/evaluate_console_output_at_epoch35.png" width="880" height="auto">
<br><br>Image-Segmentation-Curated-Spacecrafts

<a href="./projects/TensorFlowFlexUNet/Curated-Spacecrafts/evaluation.csv">evaluation.csv</a><br>
The loss (categorical_crossentropy) to this Curated-Spacecrafts/test was not  low, and dice_coef_multiclass not high as shown below.
<br>
<pre>
categorical_crossentropy,0.1696
dice_coef_multiclass,0.9286
</pre>
<br>
<h3>5 Inference</h3>
Please move to a <b>./projects/TensorFlowFlexUNet/Curated-Spacecrafts</b> folder<br>
,and run the following bat file to infer segmentation regions for images by the Trained-TensorflowFlexUNet model for Curated-Spacecrafts.<br>
<pre>
>./4.tiled_infer.bat
</pre>
This simply runs the following command.
<pre>
>python ../../../src/TensorFlowFlexUNetTiledInferencer.py ./train_eval_infer.config
</pre>
<hr>
<b>mini_test_images</b><br>
<img src="./projects/TensorFlowFlexUNet/Curated-Spacecrafts/asset/mini_test_images.png" width="1024" height="auto"><br>
<b>mini_test_mask(ground_truth)</b><br>
<img src="./projects/TensorFlowFlexUNet/Curated-Spacecrafts/asset/mini_test_masks.png" width="1024" height="auto"><br>
<hr>
<b>Inferred test masks</b><br>
<img src="./projects/TensorFlowFlexUNet/Curated-Spacecrafts/asset/mini_test_output.png" width="1024" height="auto"><br>
<br>
<hr>
<b>Enlarged images and masks for Curated-Spacecrafts Images of 1024x512 pixels</b><br>
As shown below, the inferred masks predicted by our segmentation model trained on the dataset appear similar to the ground truth masks, 
but they lack precision in certain areas, especially this model failed to detect blue regions. 
<br>
<br>
<table border=1 style='border-collapse:collapse;' cellpadding='5'>
<tr>
<th>Input: image</th>
<th>Mask (ground_truth)</th>
<th>Prediction: tiled_inferred_mask</th>
</tr>
<tr>
<td><img src="./projects/TensorFlowFlexUNet/Curated-Spacecrafts/mini_test/images/1002.png" width="320" height="auto"></td>
<td><img src="./projects/TensorFlowFlexUNet/Curated-Spacecrafts/mini_test/masks/1002.png" width="320" height="auto"></td>
<td><img src="./projects/TensorFlowFlexUNet/Curated-Spacecrafts/mini_test_output_tiled/1002.png" width="320" height="auto"></td>
</tr>

<tr>
<td><img src="./projects/TensorFlowFlexUNet/Curated-Spacecrafts/mini_test/images/1008.png" width="320" height="auto"></td>
<td><img src="./projects/TensorFlowFlexUNet/Curated-Spacecrafts/mini_test/masks/1008.png" width="320" height="auto"></td>
<td><img src="./projects/TensorFlowFlexUNet/Curated-Spacecrafts/mini_test_output_tiled/1008.png" width="320" height="auto"></td>
</tr>

<tr>
<td><img src="./projects/TensorFlowFlexUNet/Curated-Spacecrafts/mini_test/images/1010.png" width="320" height="auto"></td>
<td><img src="./projects/TensorFlowFlexUNet/Curated-Spacecrafts/mini_test/masks/1010.png" width="320" height="auto"></td>
<td><img src="./projects/TensorFlowFlexUNet/Curated-Spacecrafts/mini_test_output_tiled/1010.png" width="320" height="auto"></td>
</tr>
<tr>
<td><img src="./projects/TensorFlowFlexUNet/Curated-Spacecrafts/mini_test/images/1014.png" width="320" height="auto"></td>
<td><img src="./projects/TensorFlowFlexUNet/Curated-Spacecrafts/mini_test/masks/1014.png" width="320" height="auto"></td>
<td><img src="./projects/TensorFlowFlexUNet/Curated-Spacecrafts/mini_test_output_tiled/1014.png" width="320" height="auto"></td>
</tr>
<tr>
<td><img src="./projects/TensorFlowFlexUNet/Curated-Spacecrafts/mini_test/images/1029.png" width="320" height="auto"></td>
<td><img src="./projects/TensorFlowFlexUNet/Curated-Spacecrafts/mini_test/masks/1029.png" width="320" height="auto"></td>
<td><img src="./projects/TensorFlowFlexUNet/Curated-Spacecrafts/mini_test_output_tiled/1029.png" width="320" height="auto"></td>
</tr>
<tr>
<td><img src="./projects/TensorFlowFlexUNet/Curated-Spacecrafts/mini_test/images/1036.png" width="320" height="auto"></td>
<td><img src="./projects/TensorFlowFlexUNet/Curated-Spacecrafts/mini_test/masks/1036.png" width="320" height="auto"></td>
<td><img src="./projects/TensorFlowFlexUNet/Curated-Spacecrafts/mini_test_output_tiled/1036.png" width="320" height="auto"></td>
</tr>

</table>
<hr>
<br>
<h3>
References
</h3>
<b>1. A Spacecraft Dataset for Detection, Segmentation and Parts Recognition</b><br>
Hoang Anh Dung, Bo Chen and Tat-Jun Chin<br>
<a href="https://arxiv.org/pdf/2106.08186">
https://arxiv.org/pdf/2106.08186
</a>
<br>
<br>
<b>2. SpaceSeg: A High-Precision Intelligent Perception Segmentation Method for Multi-Spacecraft On-Orbit Targets</b><br>
Hao Liu, Pengyu Guo, Siyuan Yang, Zeqing Jiang, Qinglei Hu and Dongyu Li<br>
<a href="https://arxiv.org/pdf/2503.11133">
https://arxiv.org/pdf/2503.11133
</a>
<br>
<br>
<b>3. 3D Component Segmentation Network and Dataset for Non-Cooperative Spacecraft</b><br>
Guangyuan Zhao, Xue Wan, Yaolin, Yadong Shao andShengyang Li<br>
<a href="https://www.mdpi.com/2226-4310/9/5/248">
https://www.mdpi.com/2226-4310/9/5/248
</a>
<br>
<br>
<b>4. A New Dataset and Performance Benchmark for Real-time Spacecraft Segmentation in Onboard Flight Computerst</b><br>
Jeffrey Joan Sam, Janhavi Sathe, Nikhil Chigali, Naman Gupta, Radhey Ruparel, Yicheng Jiang, Janmajay Singh,<br>
 James W. Berck, and Arko Barman<br>
<a href="https://arxiv.org/pdf/2507.10775v1">
https://arxiv.org/pdf/2507.10775v1
</a>
<br>
<br>
<b>5. TensorFlow-FlexUNet-Image-Segmentation-Model</b><br>
Toshiyuki Arai <br>
<a href="https://github.com/sarah-antillia/TensorFlow-FlexUNet-Image-Segmentation-Model">
https://github.com/sarah-antillia/TensorFlow-FlexUNet-Image-Segmentation-Model
</a>
<br>
<br>
<b>6. TensorFlow-FlexUNet-Tiled-Image-Segmentation-Aerial-Imagery</b><br>
Toshiyuki Arai <br>
<a href="https://github.com/atlan-antillia/TensorFlow-FlexUNet-Tiled-Image-Segmentation-Aerial-Imagery">
https://github.com/atlan-antillia/TensorFlow-FlexUNet-Tiled-Image-Segmentation-Aerial-Imagery
</a>
