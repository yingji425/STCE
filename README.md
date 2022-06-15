# Spatial-temporal Concept based Explanation of 3D ConvNets
3D ACE (Automatic Concept-based Explanation) framework for interpreting 3D ConvNets.

a 3D extention to https://github.com/amiratag/ACE

Usage
===
Extract concepts and calculate importance score for each single class:

`python main.py --target_class crying --source_dir V:/ViSR-explanation --working_dir ./test/ --model_to_run keras-r3d --labels_path ./data/label.txt --bottlenecks average_pooling3d --num_random_exp 80 --max_videos 500 --min_videos 80 --imageshape 16 112 112 --model_path r3d.h5 --batchsize 60`

Paper
===
More details can be found on our extended abstract http://arxiv.org/abs/2206.05275
and poster here https://xai4cv.github.io/assets/posters/p46.pdf

Our research on 3D interpretation is still in progress.

