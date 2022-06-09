"""ACE library.

Library for discovering and testing concept activation vectors. It contains
ConceptDiscovery class that is able to discover the concepts belonging to one
of the possible classification labels of the classification task of a network
and calculate each concept's TCAV score..
"""
import sys
import os
import numpy as np
from PIL import Image
import scipy.stats as stats
import skimage.segmentation as segmentation
import sklearn.cluster as cluster
import sklearn.metrics.pairwise as metrics
import tensorflow as tf
from joblib import Parallel, delayed
from tcavvideo import cav
from ace_helpers import *
import math
from skimage.transform import resize
import pickle
import copy
import random
import multiprocessing
from tqdm import tqdm


def jaccard3D(a, b):
    """
    This will compute the Jaccard Similarity coefficient for two 3-dimensional volumes
    Volumes are expected to be of the same size. We are expecting binary masks -
    0's are treated as background and anything else is counted as data

    Arguments:
        a {Numpy array} -- 3D array with first volume
        b {Numpy array} -- 3D array with second volume
    Returns:
        float
    """
    if len(a.shape) != 3 or len(b.shape) != 3:
        raise Exception(f"Expecting 3 dimensional inputs, got {a.shape} and {b.shape}")

    if a.shape != b.shape:
        raise Exception(f"Expecting inputs of the same shape, got {a.shape} and {b.shape}")

    intersection = np.sum((a > 0) * (b > 0))
    volumes = np.sum(a > 0) + np.sum(b > 0)
    union = volumes - intersection

    return float(intersection) / float(union)


def preprocess(inputs):
    inputs[..., 0] -= 99.9
    inputs[..., 1] -= 92.1
    inputs[..., 2] -= 82.6
    inputs[..., 0] /= 65.8
    inputs[..., 1] /= 62.3
    inputs[..., 2] /= 60.3
    # inputs = np.transpose(inputs, (0, 2, 3, 1, 4))
    return inputs


class ConceptDiscovery(object):
    """Discovering and testing concepts of a class.

  For a trained network, it first discovers the concepts as areas of the images
  in the class and then calculates the TCAV score of each concept. It is also
  able to transform images from pixel space into concept space.
  """

    def __init__(self,
                 model,
                 target_class,
                 random_concept,
                 bottlenecks,
                 sess,
                 working_dir,
                 discovered_concepts_dir,
                 source_dir,
                 activation_dir,
                 cav_dir,
                 patches_dir,
                 supervoxels_dir,
                 masks_dir,
                 num_random_exp=2,
                 channel_mean=True,
                 max_videos=40,
                 min_videos=20,
                 num_discovery_videos=40,
                 num_workers=20,
                 average_image_value=117,
                 imageshape=None,
                 mode='train',
                 bs=40):
        """Runs concept discovery for a given class in a trained model.

    For a trained classification model, the ConceptDiscovery class first
    performs unsupervised concept discovery using examples of one of the classes
    in the network.

    Args:
      model: A trained classification model on which we run the concept
             discovery algorithm
      target_class: Name of the one of the classes of the network
      random_concept: A concept made of random images (used for statistical
                      test) e.g. "random500_199"
      bottlenecks: a list of bottleneck layers of the model for which the concept
                   discovery stage is performed
      sess: Model's tensorflow session
      source_dir: This directory that contains folders with images of network's
                  classes.
      activation_dir: directory to save computed activations
      cav_dir: directory to save CAVs of discovered and random concepts
      patches_dir: directory to save computed patches (original segmentation results) for each video
      supervoxels_dir: directory to save computed supervoxels (resized pathces) for each video
      num_random_exp: Number of random counterparts used for calculating several
                      CAVs and TCAVs for each concept (to make statistical
                        testing possible.)
      channel_mean: If true, for the unsupervised concept discovery the
                    bottleneck activations are averaged over channels instead
                    of using the whole acivation vector (reducing
                    dimensionality)
      max_videos: maximum number of images in a discovered concept
      min_videos : minimum number of images in a discovered concept for the
                 concept to be accepted
      num_discovery_imgs: Number of images used for concept discovery. If None,
                          will use max_imgs instead.
      num_workers: if greater than zero, runs methods in parallel with
        num_workers parallel threads. If 0, no method is run in parallel
        threads.
      average_image_value: The average value used for mean subtraction in the
                           nework's preprocessing stage.
      mode: if it is used as train or test (test mode needs to save masks as well)
    """
        self.model = model
        self.sess = sess
        self.target_class = target_class
        self.num_random_exp = num_random_exp
        if isinstance(bottlenecks, str):
            bottlenecks = [bottlenecks]
        self.bottlenecks = bottlenecks
        self.working_dir = working_dir
        self.discovered_concepts_dir = discovered_concepts_dir
        self.source_dir = source_dir
        self.activation_dir = activation_dir
        self.cav_dir = cav_dir
        self.patches_dir = patches_dir
        self.supervoxels_dir = supervoxels_dir
        self.masks_dir = masks_dir
        self.channel_mean = channel_mean
        self.random_concept = random_concept
        self.max_videos = max_videos
        self.min_videos = min_videos
        if num_discovery_videos is None:
            num_discovery_videos = max_videos
        self.num_discovery_videos = num_discovery_videos
        self.num_workers = num_workers
        self.average_image_value = average_image_value
        self.image_shape = [int(i) for i in imageshape]
        self.mode = mode
        self.bs = bs

    def load_concept_videos(self, concept, max_videos=100, do_shuffle=True):
        """Loads all colored videos of a concept.

    Args:
      concept: The name of the concept to be loaded
      max_videos: maximum number of videos to be loaded
      do_shuffle: if shuffle video list

    Returns:
      video paths of the desired concept or class.
    """
        concept_dir = os.path.join(self.source_dir, concept)
        video_paths = [
            os.path.join(concept_dir, d)
            for d in tf.io.gfile.listdir(concept_dir)
        ]
        return get_video_list(video_paths, max_videos, do_shuffle)

    def create_patches(self, discovery_videos=None,
                       param_dict=None):
        """Creates a set of video patches using supervoxel methods.

    This method takes in the concept discovery images and transforms it to a
    dataset made of the patches of those videos.

    Args:
      discovery_videos: videos used for creating patches. If None, the videos in
        the target class folder are used.

      param_dict: Contains parameters of the superpixel method used in the form
                of {'param1':[a,b,...], 'param2':[z,y,x,...], ...}. For instance
                {'n_segments':[150,500,800], 'compactness':[100,100,100]} for slic
                method.
    """
        if tf.io.gfile.exists(self.supervoxels_dir) and os.listdir(self.supervoxels_dir):
            with open(self.working_dir + '/' + 'raw_videos_list.txt', 'r') as f:
                raw_videos_list = [line.rstrip('\n') for line in f]
            self.discovery_videos = raw_videos_list
            with open(self.working_dir + '/' + 'patch_info.pkl', 'rb') as f:
                patch_info = pickle.load(f)
            if self.mode == 'train':
                self.dataset, self.video_numbers, self.patches, self.index = \
                    patch_info[0], patch_info[1], patch_info[2], patch_info[3]
            else:
                self.dataset, self.video_numbers, self.masks = \
                    patch_info[0], patch_info[1], patch_info[2]

        else:
            if param_dict is None:
                param_dict = {}
            dataset, video_numbers, patches = [], [], []
            if discovery_videos is None:
                if self.mode == 'train':
                    raw_videos_list = self.load_concept_videos(
                        self.target_class, self.num_discovery_videos)
                    self.discovery_videos = raw_videos_list
                    with open(self.working_dir + '/' + 'raw_videos_list.txt', 'w') as output:
                        output.writelines("%s\n" % video for video in raw_videos_list)
                else:
                    raw_videos_list = self.load_concept_videos(
                        self.target_class, self.num_discovery_videos, do_shuffle=False)
                    self.discovery_videos = raw_videos_list
                    with open(self.working_dir + '/' + 'raw_videos_list.txt', 'w') as output:
                        output.writelines("%s\n" % video for video in raw_videos_list)
            else:
                # self.discovery_videos: videos in target class
                self.discovery_videos = discovery_videos

            if self.mode == 'train':
                # in train stage, mask information don't need to save
                supervoxel_paths = []
                patch_paths = []
                index_all = []
                for fn, video in enumerate(self.discovery_videos):
                    # laod video as size [112, 112] range from [0,255]
                    video = load_video_from_file(video, self.image_shape)
                    video_dir = os.path.join(self.discovered_concepts_dir, 'videos')
                    tf.io.gfile.makedirs(video_dir)
                    save_video(fn, video_dir, video.astype(np.uint8))
                    # build folder for each video to save supervoxels and patches
                    patches_dir = os.path.join(self.patches_dir, '0' * (3 - int(np.log10(fn + 1))) + str(fn + 1))
                    supervoxels_dir = os.path.join(self.supervoxels_dir,
                                                   '0' * (3 - int(np.log10(fn + 1))) + str(fn + 1))
                    tf.io.gfile.makedirs(patches_dir)
                    tf.io.gfile.makedirs(supervoxels_dir)
                    param_dict_c = param_dict.copy()
                    video_supervoxels, video_patches, final_mask, index = self._return_supervoxels(
                        video, param_dict_c)
                    final_mask.clear()
                    i = 0
                    for c in range(len(video_patches)):
                        if video_patches[c].shape[0] >= 16:
                            supervoxel_path = (os.path.join(supervoxels_dir, str(i) + '.pkl'))
                            self.save_patch(video_supervoxels[c], supervoxel_path)
                            supervoxel_paths.append(supervoxel_path)

                            patch_path = (os.path.join(patches_dir, str(i) + '.pkl'))
                            self.save_patch(video_patches[c], patch_path)
                            patch_paths.append(patch_path)

                            index_all.append(index[c])
                            video_numbers.append(fn)
                            i = i + 1
                            print('saving', i, 'th patches for video ', fn)
                    # Saving the concept discovery target class images
                    video_supervoxels.clear()
                    video_patches.clear()
                    del video
                    print(fn)
                self.dataset, self.video_numbers, self.patches, self.index = \
                    supervoxel_paths, np.array(video_numbers), patch_paths, index_all
                patch_info = [supervoxel_paths, np.array(video_numbers), patch_paths, index_all]
                f = open(self.working_dir + '/' + 'patch_info.pkl', "wb")
                pickle.dump(patch_info, f)
                f.close()

            elif self.mode == 'test':
                # in test stage, patch information don't need to save
                supervoxel_paths = []
                # patch_paths = []
                mask_paths = []
                for fn, video in enumerate(self.discovery_videos):
                    video = load_video_from_file(video, self.image_shape)
                    video_dir = os.path.join(self.discovered_concepts_dir, 'videos')
                    tf.io.gfile.makedirs(video_dir)
                    save_video(fn, video_dir, video.astype(np.uint8))
                    # build folder for each video to save supervoxels and patches
                    supervoxels_dir = os.path.join(self.supervoxels_dir,
                                                   '0' * (3 - int(np.log10(fn + 1))) + str(fn + 1))
                    masks_dir = os.path.join(self.masks_dir, '0' * (3 - int(np.log10(fn + 1))) + str(fn + 1))
                    tf.io.gfile.makedirs(supervoxels_dir)
                    tf.io.gfile.makedirs(masks_dir)
                    param_dict_c = param_dict.copy()
                    video_supervoxels, video_patches, final_mask = self._return_supervoxels_test(
                        video, param_dict_c)
                    video_patches.clear()
                    # print(np.min(np.add.reduce(final_mask)))

                    i = 0
                    for c in range(len(video_supervoxels)):
                        if video_supervoxels[c].shape[0] >= 16:
                            supervoxel_path = (os.path.join(supervoxels_dir, str(i) + '.pkl'))
                            self.save_patch(video_supervoxels[c], supervoxel_path)
                            supervoxel_paths.append(supervoxel_path)

                            mask_path = (os.path.join(masks_dir, str(i) + '.pkl'))
                            self.save_patch(final_mask[c], mask_path)
                            mask_paths.append(mask_path)
                            video_numbers.append(fn)
                            i = i+1
                    # Saving the concept discovery target class images
                    video_supervoxels.clear()
                    final_mask.clear()
                    del video
                    print(fn)
                self.dataset, self.video_numbers, self.masks = \
                    supervoxel_paths, np.array(video_numbers), mask_paths
                patch_info = [supervoxel_paths, np.array(video_numbers), mask_paths]
                f = open(self.working_dir + '/' + 'patch_info.pkl', "wb")
                pickle.dump(patch_info, f)
                f.close()

    @staticmethod
    def save_patch(patch, patch_path):
        """save calculated patches to file(pickle file).

        Args:
            patch: patches or voxels to be saved
            patch_path: the location of the saved patches
        """
        if patch_path is not None:
            with tf.io.gfile.GFile(patch_path, 'w') as pkl_file:
                pickle.dump(patch, pkl_file)
        else:
            tf.compat.v1.logging.info('save_path is None. Not saving anything')

    def _return_supervoxels(self, video, param_dict=None):
        """Returns all patches for one video.

    Given an video as size of [frames, height, width, channel], calculates supervoxels for each of the parameter lists
    in param_dict and returns a set of unique supervoxels by removing duplicates.
    If two patches have Jaccard similarity more than 0.5,
    they are considered duplicates.

    Args:
      video: The input video
      method: supervoxel method, here we only use slic
      param_dict: Contains parameters of the supervoxel method used in the form
                of {'param1':[a,b,...], 'param2':[z,y,x,...], ...}. For instance
                {'n_segments':[150,500,800], 'compactness':[100,100,100]} for slic
                method.
    Raises:
      ValueError: if the segmentation method is invalid.
    """
        if param_dict is None:
            param_dict = {}
        n_segmentss = param_dict.pop('n_segments', [15, 50, 80])
        n_params = len(n_segmentss)
        compactnesses = param_dict.pop('compactness', [20] * n_params)
        sigmas = param_dict.pop('sigma', [1.] * n_params)
        unique_masks = []
        unique_index = []
        unique_masks_save = []
        for i in range(n_params):
            # param_index is the start and end index of the voxel
            param_masks = []
            param_index = []
            mask_save = []
            segments = segmentation.slic(
                video, n_segments=n_segmentss[i], compactness=compactnesses[i],
                sigma=sigmas[i])
            for s in range(segments.max()+1):
                mask_all = (segments == s).astype(float)
                # find where the corresponding segment start and end
                # this mask is a 3D voxel with depth, height, width
                if np.where(mask_all == 1)[0].size==0:
                    continue
                else:
                    st, end = min(np.where(mask_all == 1)[0]), max(np.where(mask_all == 1)[0]) + 1
                    mask = mask_all[st:end, :, :]
                    # does this threshold need to be change?
                    if np.mean(mask) > 0.001:
                        unique = True
                        for seen_mask in unique_masks:
                            # compare the similarity of 3D segments, so we need to resize the voxels to the same size
                            if mask.shape[0] < seen_mask.shape[0]:
                                resize_shape = mask.shape[0]
                                start_ind = math.floor((seen_mask.shape[0] - resize_shape) / 2)
                                end_ind = start_ind + resize_shape
                                seen_mask_com = seen_mask[start_ind:end_ind, :, :].copy()
                                mask_com = mask.copy()
                            else:
                                resize_shape = seen_mask.shape[0]
                                start_ind = math.floor((mask.shape[0] - resize_shape) / 2)
                                end_ind = start_ind + resize_shape
                                mask_com = mask[start_ind:end_ind, :, :].copy()
                                seen_mask_com = seen_mask.copy()
                            jaccard = jaccard3D(seen_mask_com, mask_com)
                            if jaccard > 0.5:
                                unique = False
                                break
                        # after comparison, the mask we want to save is still original size.
                        # so we use seen_mask_com and mask_com only for comparison
                        if unique:
                            param_masks.append(mask)
                            param_index.append([st, end])
                            mask_save.append(mask_all)
            unique_masks.extend(param_masks)
            unique_index.extend(param_index)
            unique_masks_save.extend(mask_save)
        index = copy.deepcopy(unique_index)
        index.reverse()
        supervoxels, patches = [], []
        while unique_masks:
            supervoxel, patch = self._extract_patch(video, unique_masks.pop(), unique_index.pop())
            supervoxels.append(supervoxel)
            patches.append(patch)
        # since we use pop, so the index list should be reversed
        return supervoxels, patches, unique_masks_save, index

    def _return_supervoxels_test(self, video, param_dict=None):
        """Returns all patches for one video.

    Given an video as size of [frames, height, width, channel], calculates supervoxels for each of the parameter lists
    in param_dict and returns a set of unique supervoxels by removing duplicates.
    If two patches have Jaccard similarity more than 0.5,
    they are considered duplicates.

    Args:
      video: The input video
      method: supervoxel method, here we only use slic
      param_dict: Contains parameters of the supervoxel method used in the form
                of {'param1':[a,b,...], 'param2':[z,y,x,...], ...}. For instance
                {'n_segments':[150,500,800], 'compactness':[100,100,100]} for slic
                method.
    Raises:
      ValueError: if the segmentation method is invalid.
    """
        if param_dict is None:
            param_dict = {}
        n_segmentss = param_dict.pop('n_segments', [15, 50, 80])
        n_params = len(n_segmentss)
        compactnesses = param_dict.pop('compactness', [20] * n_params)
        sigmas = param_dict.pop('sigma', [1.] * n_params)
        unique_masks = []
        unique_index = []
        unique_masks_save = []
        for i in range(n_params):
            # param_index is the start and end index of the voxel
            param_masks = []
            param_index = []
            mask_save = []
            segments = segmentation.slic(
                video, n_segments=n_segmentss[i], compactness=compactnesses[i],
                sigma=sigmas[i])

            for s in range(segments.max()+1):
                mask_all = (segments == s).astype(float)
                # find where the corresponding segment start and end
                # this mask is a 3D voxel with depth, height, width
                st, end = min(np.where(mask_all == 1)[0]), max(np.where(mask_all == 1)[0]) + 1
                mask = mask_all[st:end, :, :]
                param_masks.append(mask)
                param_index.append([st, end])
                mask_save.append(mask_all)
            unique_masks.extend(param_masks)
            unique_index.extend(param_index)
            unique_masks_save.extend(mask_save)
        supervoxels, patches = [], []
        while unique_masks:
            supervoxel, patch = self._extract_patch(self, video, unique_masks.pop(), unique_index.pop())
            supervoxels.append(supervoxel)
            patches.append(patch)
        return supervoxels, patches, unique_masks_save


    def _return_supervoxels_no_patch(self, video, param_dict=None):
        """Returns all supervoxels for one video, do not return patches

    Given an video as size of [frames, height, width, channel], calculates supervoxels for each of the parameter lists
    in param_dict and returns a set of unique supervoxels by removing duplicates.
    If two patches have Jaccard similarity more than 0.5,
    they are considered duplicates.

    Args:
      video: The input video
      method: supervoxel method, here we only use slic
      param_dict: Contains parameters of the supervoxel method used in the form
                of {'param1':[a,b,...], 'param2':[z,y,x,...], ...}. For instance
                {'n_segments':[150,500,800], 'compactness':[100,100,100]} for slic
                method.
    Raises:
      ValueError: if the segmentation method is invalid.
    """
        if param_dict is None:
            param_dict = {}
        n_segmentss = param_dict.pop('n_segments', [15, 50, 80])
        n_params = len(n_segmentss)
        compactnesses = param_dict.pop('compactness', [20] * n_params)
        sigmas = param_dict.pop('sigma', [1.] * n_params)
        unique_masks = []
        unique_index = []
        unique_masks_save = []
        for i in range(n_params):
            # param_index is the start and end index of the voxel
            param_masks = []
            param_index = []
            mask_save = []
            segments = segmentation.slic(
                video, n_segments=n_segmentss[i], compactness=compactnesses[i],
                sigma=sigmas[i])

            for s in range(segments.max()+1):
                mask_all = (segments == s).astype(float)
                # find where the corresponding segment start and end
                # this mask is a 3D voxel with depth, height, width
                st, end = min(np.where(mask_all == 1)[0]), max(np.where(mask_all == 1)[0]) + 1
                mask = mask_all[st:end, :, :]
                param_masks.append(mask)
                param_index.append([st, end])
                mask_save.append(mask_all)
            unique_masks.extend(param_masks)
            unique_index.extend(param_index)
            unique_masks_save.extend(mask_save)
        supervoxels = []
        while unique_masks:
            supervoxel = self._extract_patch_no_patch(self, video, unique_masks.pop(), unique_index.pop())
            supervoxels.append(supervoxel)
            # patches.append(patch)
        return supervoxels, unique_masks_save


    def get_batch(self, input_pairs, liFiles):
        batch_size = len(input_pairs)
        batch = np.zeros((batch_size, self.image_shape[0], self.image_shape[1], self.image_shape[2], 3), dtype=np.float32)
        for i in range(batch_size):
            target_index = input_pairs[i]
            liTarget = liFiles[target_index[0]: target_index[1] + 1]
            batch[i][:][:][:][:] = liTarget
        batch = preprocess(batch)
        return batch

    def _extract_patch(self, video, mask, index):
        """Extracts a patch out of a video.

    Args:
      video: The original image
      mask: The binary mask of the patch area
      index: for each mask, it may only be a sub part of video. so the start and
        end index of each mask have been recorded

    Returns:
      video_resized: The resized patch such that its boundaries touches the
        video boundaries
      patch: The original patch. Rest of the video is padded with average value
    """
        patch = []
        resized_video = []
        ones = np.where(mask == 1)
        h1, h2, w1, w2 = ones[1].min(), ones[1].max(), ones[2].min(), ones[2].max()
        st, end = index[0], index[1]
        video_small = video[st:end, :, :]
        for i in range(len(mask)):
            single_mask = mask[i]
            single_frame = video_small[i]
            single_mask_expanded = np.expand_dims(single_mask, -1)
            # mask part maintain the original pixel and other part padding with average value
            # single_image = (single_mask_expanded * single_frame + (
            #         1 - single_mask_expanded) * float(self.average_image_value) / 255)
            single_image = (single_mask_expanded * single_frame + (
                    1 - single_mask_expanded) * self.average_image_value)
            patch.append(single_image)
            # resize each frame to small size according to the mask boundary
            # image = Image.fromarray((single_image[h1:h2, w1:w2] * 255).astype(np.uint8))
            # image_resized = np.array(image.resize((single_frame.shape[1], single_frame.shape[0]),
            #                                       Image.BICUBIC)).astype(float) / 255
            image = Image.fromarray(single_image[h1:h2, w1:w2].astype(np.uint8))
            image_resized = np.array(image.resize((single_frame.shape[1], single_frame.shape[0]),
                                                  Image.BICUBIC)).astype(float)
            resized_video.append(image_resized)
        resized_video = np.array(resized_video)
        patch = np.array(patch)
        return np.array(resized_video), np.array(patch)


    def _extract_patch_no_patch(self, video, mask, index):
        """Extracts a patch out of a video. not return patch

    Args:
      video: The original image
      mask: The binary mask of the patch area
      index: for each mask, it may only be a sub part of video. so the start and
        end index of each mask have been recorded

    Returns:
      video_resized: The resized patch such that its boundaries touches the
        video boundaries
      patch: The original patch. Rest of the video is padded with average value
    """
        resized_video = []
        ones = np.where(mask == 1)
        h1, h2, w1, w2 = ones[1].min(), ones[1].max(), ones[2].min(), ones[2].max()
        st, end = index[0], index[1]
        video_small = video[st:end, :, :]
        for i in range(len(mask)):
            single_mask = mask[i]
            single_frame = video_small[i]
            single_mask_expanded = np.expand_dims(single_mask, -1)
            # mask part maintain the original pixel and other part padding with average value
            # single_image = (single_mask_expanded * single_frame + (
            #         1 - single_mask_expanded) * float(self.average_image_value) / 255)
            single_image = (single_mask_expanded * single_frame + (
                    1 - single_mask_expanded) * self.average_image_value)
            # resize each frame to small size according to the mask boundary
            # image = Image.fromarray((single_image[h1:h2, w1:w2] * 255).astype(np.uint8))
            # image_resized = np.array(image.resize((single_frame.shape[1], single_frame.shape[0]),
            #                                       Image.BICUBIC)).astype(float) / 255
            image = Image.fromarray(single_image[h1:h2, w1:w2].astype(np.uint8))
            image_resized = np.array(image.resize((single_frame.shape[1], single_frame.shape[0]),
                                                  Image.BICUBIC)).astype(float)
            resized_video.append(image_resized)
        resized_video = np.array(resized_video)
        return np.array(resized_video)

    def _patch_activations(self, videos, bottleneck, bs=20, channel_mean=None):
        """Returns activations of a list of imgs.

    Args:
      imgs: List/array of images to calculate the activations of
      bottleneck: Name of the bottleneck layer of the model where activations
        are calculated
      bs: The batch size for calculating activations. (To control computational
        cost)
      channel_mean: If true, the activations are averaged across channel.

    Returns:
      The array of activations
    """

        if channel_mean is None:
            channel_mean = self.channel_mean
        if self.num_workers:
            pool = multiprocessing.Pool(self.num_workers)
            output = pool.map(
                lambda i: self.model.run_examples(videos[i * bs:(i + 1) * bs], bottleneck),
                np.arange(int(videos.shape[0] / bs) + 1))
        else:
            output = []
            for i in range(int(math.ceil(len(videos) / bs))):
                output.append(
                    self.model.run_examples(videos[i * bs:(i + 1) * bs], bottleneck))
        output = np.concatenate(output, 0)
        if channel_mean and len(output.shape) > 3:
            output = np.mean(output, (1, 2))
        else:
            output = np.reshape(output, [output.shape[0], -1])
        return output

    def one_video_activation(self, video, bottleneck, channel_mean=None):
        """Returns activation for one video.

    Args:
      video: the video to calculate the activations of
      bottleneck: Name of the bottleneck layer of the model where activations
        are calculated
      channel_mean: If true, the activations are averaged across channel.

    Returns:
      The array of activation
    """

        if channel_mean is None:
            channel_mean = self.channel_mean
        output = self.model.run_examples(video, bottleneck)
        output = np.concatenate(output, 0)
        if channel_mean and len(output.shape) > 3:
            output = np.mean(output, (1, 2))
        else:
            output = np.reshape(output, [output.shape[0], -1])
        return output

    def _cluster(self, acts, method='KM', param_dict=None):
        """Runs unsupervised clustering algorithm on concept actiavtations.

    Args:
      acts: activation vectors of datapoints points in the bottleneck layer.
        E.g. (number of clusters,) for Kmeans
      method: clustering method. We have:
        'KM': Kmeans Clustering
        'AP': Affinity Propagation
        'SC': Spectral Clustering
        'MS': Mean Shift clustering
        'DB': DBSCAN clustering method
      param_dict: Contains superpixl method's parameters. If an empty dict is
                 given, default parameters are used.

    Returns:
      asg: The cluster assignment label of each data points
      cost: The clustering cost of each data point
      centers: The cluster centers. For methods like Affinity Propagetion
      where they do not return a cluster center or a clustering cost, it
      calculates the medoid as the center  and returns distance to center as
      each data points clustering cost.

    Raises:
      ValueError: if the clustering method is invalid.
    """
        if param_dict is None:
            param_dict = {}
        centers = None
        if method == 'KM':
            n_clusters = param_dict.pop('n_clusters', 25)
            km = cluster.KMeans(n_clusters)
            d = km.fit(acts)
            centers = km.cluster_centers_
            # L2 normalization
            d = np.linalg.norm(
                np.expand_dims(acts, 1) - np.expand_dims(centers, 0), ord=2, axis=-1)
            asg, cost = np.argmin(d, -1), np.min(d, -1)
        elif method == 'AP':
            damping = param_dict.pop('damping', 0.5)
            ca = cluster.AffinityPropagation(damping)
            ca.fit(acts)
            centers = ca.cluster_centers_
            d = np.linalg.norm(
                np.expand_dims(acts, 1) - np.expand_dims(centers, 0), ord=2, axis=-1)
            asg, cost = np.argmin(d, -1), np.min(d, -1)
        elif method == 'MS':
            ms = cluster.MeanShift(n_jobs=self.num_workers)
            asg = ms.fit_predict(acts)
        elif method == 'SC':
            n_clusters = param_dict.pop('n_clusters', 25)
            sc = cluster.SpectralClustering(
                n_clusters=n_clusters, n_jobs=self.num_workers)
            asg = sc.fit_predict(acts)
        elif method == 'DB':
            eps = param_dict.pop('eps', 0.5)
            min_samples = param_dict.pop('min_samples', 20)
            sc = cluster.DBSCAN(eps, min_samples, n_jobs=self.num_workers)
            asg = sc.fit_predict(acts)
        else:
            raise ValueError('Invalid Clustering Method!')
        if centers is None:  ## If clustering returned cluster centers, use medoids
            centers = np.zeros((asg.max() + 1, acts.shape[1]))
            cost = np.zeros(len(acts))
            for cluster_label in range(asg.max() + 1):
                cluster_idxs = np.where(asg == cluster_label)[0]
                cluster_points = acts[cluster_idxs]
                pw_distances = metrics.euclidean_distances(cluster_points)
                centers[cluster_label] = cluster_points[np.argmin(
                    np.sum(pw_distances, -1))]
                cost[cluster_idxs] = np.linalg.norm(
                    acts[cluster_idxs] - np.expand_dims(centers[cluster_label], 0),
                    ord=2,
                    axis=-1)
        return asg, cost, centers

    def video3D(self, file):
        depth = int(self.image_shape[0])
        nframe = file.shape[0]
        frame = [math.floor(x * nframe / depth) for x in range(depth)]
        frames = file[frame]
        input_video = []
        for single_frame in frames:
            s = resize(single_frame, (int(self.image_shape[1]), int(self.image_shape[2])))
            input_video.append(s)
        return np.array(input_video)

    def process_video_batch(self, input_video, bs, bn):
        acts = []
        frame_num = len(input_video)
        start_indexes = list(range(0, frame_num - self.image_shape[0] + 1))
        end_indexes = list(range(0 + self.image_shape[0] - 1, frame_num))
        pairs = list(zip(start_indexes, end_indexes))
        for p in range(int(math.ceil(len(pairs) / bs))):
            input_pairs = pairs[p * bs:(p + 1) * bs]
            batch_voxel = self.get_batch(input_pairs, input_video)
            acts.append(get_acts_from_videos(batch_voxel, self.model, bn))
        return np.concatenate(acts, 0)

    def discover_concepts(self,
                          method='KM',
                          activations=None,
                          param_dicts=None,
                          ):
        """Discovers the frequent occurring concepts in the target class.

      Calculates self.dic, a dicationary containing all the informations of the
      discovered concepts in the form of {'bottleneck layer name: bn_dic} where
      bn_dic itself is in the form of {'concepts:list of concepts,
      'concept name': concept_dic} where the concept_dic is in the form of
      {'images': resized patches of concept, 'patches': original patches of the
      concepts, 'image_numbers': image id of each patch}

    Args:
      method: Clustering method.
      activations: If activations are already calculated. If not calculates
                   them. Must be a dictionary in the form of {'bn':array, ...}
      param_dicts: A dictionary in the format of {'bottleneck':param_dict,...}
                   where param_dict contains the clustering method's parametrs
                   in the form of {'param1':value, ...}. For instance for Kmeans
                   {'n_clusters':25}. param_dicts can also be in the format
                   of param_dict where same parameters are used for all
                   bottlenecks.
    """
        if param_dicts is None:
            param_dicts = {}
        if set(param_dicts.keys()) != set(self.bottlenecks):
            param_dicts = {bn: param_dicts for bn in self.bottlenecks}
        self.dic = {}  # The main dictionary of the ConceptDiscovery class.
        for bn in self.bottlenecks:
            if os.path.exists(self.working_dir + '/' + 'dic.pkl'):
                with open(self.working_dir + '/' + 'dic.pkl', 'rb') as f:
                    self.dic[bn] = pickle.load(f)
            else:
                bn_dic = {}
                bn_activations = []
                if os.path.exists(self.working_dir + '/' + 'activation.pkl'):
                    with open(self.working_dir + '/' + 'activation.pkl', 'rb') as f:
                        bn_activations = pickle.load(f)
                else:
                    if activations is None or bn not in activations.keys():
                        for single_voxel_path in tqdm(self.dataset):
                            input_video = pickle.load(open(single_voxel_path, 'rb'))
                            activation = np.mean(self.process_video_batch(input_video, self.bs, bn), 0)
                            if activation.ndim > 3:
                                activation = activation.flatten()
                            bn_activations.append(activation)
                        bn_activations = np.array(bn_activations)
                        f = open(self.working_dir + '/' + 'activation.pkl', "wb")
                        pickle.dump(bn_activations, f)
                        f.close()
                    else:
                        bn_activations = activations[bn]
                bn_dic['label'], bn_dic['cost'], centers = self._cluster(
                    bn_activations, method, param_dicts[bn])
                concept_number, bn_dic['concepts'] = 0, []
                for i in range(bn_dic['label'].max() + 1):
                    label_idxs = np.where(bn_dic['label'] == i)[0]
                    if len(label_idxs) > self.min_videos:
                        concept_costs = bn_dic['cost'][label_idxs]
                        # find the max_imgs number nearest images
                        concept_idxs = label_idxs[np.argsort(concept_costs)[:self.max_videos]]
                        concept_video_numbers = set(self.video_numbers[label_idxs])
                        discovery_size = len(self.discovery_videos)
                        """
                        make judgement if the images are from almost the same discovery images
                        unpopular clusters that have only few members
                        """
                        highly_common_concept = len(
                            concept_video_numbers) > 0.4 * len(label_idxs)
                        mildly_common_concept = len(
                            concept_video_numbers) > 0.1 * len(label_idxs)
                        mildly_populated_concept = len(
                            concept_video_numbers) > 0.1 * discovery_size
                        cond2 = mildly_populated_concept and mildly_common_concept
                        non_common_concept = len(
                            concept_video_numbers) > 0.08 * len(label_idxs)
                        highly_populated_concept = len(
                            concept_video_numbers) > 0.35 * discovery_size
                        cond3 = non_common_concept and highly_populated_concept
                        if highly_common_concept or cond2 or cond3:
                            concept_number += 1
                            concept = '{}_concept{}'.format(self.target_class, concept_number)
                            bn_dic['concepts'].append(concept)
                            bn_dic[concept] = {
                                'videos': [self.dataset[index] for index in concept_idxs],
                                'patches': [self.patches[index] for index in concept_idxs],
                                'activations': bn_activations[concept_idxs],
                                'video_numbers': self.video_numbers[concept_idxs],
                                'concept_index': [self.index[index] for index in concept_idxs]
                            }
                            bn_dic[concept + '_center'] = centers[i]
                bn_dic.pop('label', None)
                bn_dic.pop('cost', None)
                self.dic[bn] = bn_dic
                f = open(self.working_dir + '/' + 'dic.pkl', "wb")
                pickle.dump(bn_dic, f)
                f.close()

    def _random_concept_activations(self, bottleneck, random_concept):
        """Wrapper for computing or loading activations of random concepts.

    Takes care of making, caching (if desired) and loading activations.

    Args:
      bottleneck: The bottleneck layer name
      random_concept: Name of the random concept e.g. "random500_0"

    Returns:
      A nested dict in the form of {concept:{bottleneck:activation}}
    """
        rnd_acts_path = os.path.join(self.activation_dir, 'acts_{}_{}'.format(
            random_concept, bottleneck))
        if not tf.io.gfile.exists(rnd_acts_path):
            acts = []
            rnd_videos_list = self.load_concept_videos(random_concept, self.max_videos)  # get random video list
            for single_video_path in rnd_videos_list:
                single_video = load_video_from_file(single_video_path, self.image_shape)
                activation = np.mean(self.process_video_batch(single_video, self.bs, bottleneck), 0)
                acts.append(activation)
            acts = np.array(acts)
            with open(rnd_acts_path, 'wb') as f:
                np.save(f, acts, allow_pickle=False)
            del acts
        return np.load(rnd_acts_path).squeeze()

    def _calculate_cav(self, c, r, bn, act_c, ow, directory=None):
        """Calculates a sinle cav for a concept and a one random counterpart.

    Args:
      c: conept name
      r: random concept name
      bn: the layer name
      act_c: activation matrix of the concept in the 'bn' layer
      ow: overwrite if CAV already exists
      directory: to save the generated CAV

    Returns:
      The accuracy of the CAV
    """
        if directory is None:
            directory = self.cav_dir
        act_r = self._random_concept_activations(bn, r)
        cav_instance = cav.get_or_train_cav([c, r],
                                            bn, {
                                                c: {
                                                    bn: act_c
                                                },
                                                r: {
                                                    bn: act_r
                                                }
                                            },
                                            cav_dir=directory,
                                            overwrite=ow)
        return cav_instance.accuracies['overall']

    def _concept_cavs(self, bn, concept, activations, randoms=None, ow=True):
        """Calculates CAVs of a concept versus all the random counterparts.

    Args:
      bn: bottleneck layer name
      concept: the concept name
      activations: activations of the concept in the bottleneck layer
      randoms: None if the class random concepts are going to be used
      ow: If true, overwrites the existing CAVs

    Returns:
      A dict of cav accuracies in the form of {'bottleneck layer':
      {'concept name':[list of accuracies], ...}, ...}
    """
        if randoms is None:
            randoms = [
                'random500_{}'.format(i) for i in np.arange(self.num_random_exp)
            ]
        if self.num_workers:
            pool = multiprocessing.Pool(20)
            accs = pool.map(
                lambda rnd: self._calculate_cav(concept, rnd, bn, activations, ow),
                randoms)
        else:
            accs = []
            for rnd in randoms:
                accs.append(self._calculate_cav(concept, rnd, bn, activations, ow))
                print("calculate random activations for random {}".format(rnd))
        return accs

    def cavs(self, min_acc=0., ow=True):
        """Calculates cavs for all discovered concepts.

    This method calculates and saves CAVs for all the discovered concepts
    versus all random concepts in all the bottleneck layers

    Args:
      min_acc: Delete discovered concept if the average classification accuracy
        of the CAV is less than min_acc
      ow: If True, overwrites an already calcualted cav.

    Returns:
      A dicationary of classification accuracy of linear boundaries orthogonal
      to cav vectors
    """
        acc = {bn: {} for bn in self.bottlenecks}
        concepts_to_delete = []
        for bn in self.bottlenecks:
            for concept in self.dic[bn]['concepts']:
                concept_acts = []
                for video_path in self.dic[bn][concept]['videos']:
                    concept_video = pickle.load(open(video_path, 'rb'))
                    activation = np.mean(self.process_video_batch(concept_video, self.bs, bn), 0)
                    concept_acts.append(activation)
                concept_acts = np.array(concept_acts)
                acc[bn][concept] = self._concept_cavs(bn, concept, concept_acts, ow=ow)
                if np.mean(acc[bn][concept]) < min_acc:
                    concepts_to_delete.append((bn, concept))
            target_class_acts = []
            for video_path in self.discovery_videos:
                single_video = load_video_from_file(video_path, self.image_shape)
                # print(single_video.shape)
                activation = np.mean(self.process_video_batch(single_video, self.bs, bn), 0)
                target_class_acts.append(activation)
            target_class_acts = np.array(target_class_acts)
            acc[bn][self.target_class] = self._concept_cavs(
                bn, self.target_class, target_class_acts, ow=ow)
            rnd_acts = self._random_concept_activations(bn, self.random_concept)
            acc[bn][self.random_concept] = self._concept_cavs(
                bn, self.random_concept, rnd_acts, ow=ow)
        for bn, concept in concepts_to_delete:
            self.delete_concept(bn, concept)
        return acc

    def load_cav_direction(self, c, r, bn, directory=None):
        """Loads an already computed cav.

    Args:
      c: concept name
      r: random concept name
      bn: bottleneck layer
      directory: where CAV is saved

    Returns:
      The cav instance
    """
        if directory is None:
            directory = self.cav_dir
        params = {'model_type': 'linear', 'alpha': .01}
        cav_key = cav.CAV.cav_key([c, r], bn, params['model_type'], params['alpha'])
        cav_path = os.path.join(self.cav_dir, cav_key.replace('/', '.') + '.pkl')
        vector = cav.CAV.load_cav(cav_path).cavs[0]
        return np.expand_dims(vector, 0) / np.linalg.norm(vector, ord=2)

    def _sort_concepts(self, scores):
        for bn in self.bottlenecks:
            tcavs = []
            for concept in self.dic[bn]['concepts']:
                tcavs.append(np.mean(scores[bn][concept]))
            concepts = []
            for idx in np.argsort(tcavs)[::-1]:
                concepts.append(self.dic[bn]['concepts'][idx])
            self.dic[bn]['concepts'] = concepts

    def _return_gradients(self, videos):
        """For the given images calculates the gradient tensors.

    Args:
      images: Images for which we want to calculate gradients.

    Returns:
      A dictionary of images gradients in all bottleneck layers.
    """

        gradients = {}
        # class_id = self.model.label_to_id(self.target_class.replace('_', ' '))
        class_id = self.model.label_to_id(self.target_class)
        for bn in self.bottlenecks:
            acts = get_acts_from_videos(videos, self.model, bn)
            bn_grads = np.zeros((acts.shape[0], np.prod(acts.shape[1:])))
            for i in range(len(acts)):
                bn_grads[i] = self.model.get_gradient(
                    acts[i:i + 1], [class_id], bn, None).reshape(-1)
            gradients[bn] = bn_grads
        return gradients

    def _return_gradients_single(self, videos_list):
        """For the given video_list calculates the gradient tensors.
        calculate act for video one by one

    Args:
      videos_list: video path for which we want to calculate gradients.

    Returns:
      A dictionary of images gradients in all bottleneck layers.
    """
        gradients = {}
        # class_id = self.model.label_to_id(self.target_class.replace('_', ' '))
        class_id = self.model.label_to_id(self.target_class)
        for bn in self.bottlenecks:
            acts = []
            for single_video_path in videos_list:
                single_video = load_video_from_file(single_video_path, self.image_shape)
                activation = np.mean(self.process_video_batch(single_video, self.bs, bn), 0)
                acts.append(activation)
            acts = np.array(acts)
            bn_grads = np.zeros((acts.shape[0], np.prod(acts.shape[1:])))
            for i in range(len(acts)):
                bn_grads[i] = self.model.get_gradient(
                    acts[i:i + 1], [class_id], bn).reshape(-1)
            gradients[bn] = bn_grads
        return gradients

    def _tcav_score(self, bn, concept, rnd, gradients):
        """Calculates and returns the TCAV score of a concept.

    Args:
      bn: bottleneck layer
      concept: concept name
      rnd: random counterpart
      gradients: Dict of gradients of tcav_score_images

    Returns:
      TCAV score of the concept with respect to the given random counterpart
    """
        vector = self.load_cav_direction(concept, rnd, bn)
        prod = np.sum(gradients[bn] * vector, -1)
        return np.mean(prod < 0)

    def tcavs(self, test=False, sort=True, tcav_score_videos=None):
        """Calculates TCAV scores for all discovered concepts and sorts concepts.

    This method calculates TCAV scores of all the discovered concepts for
    the target class using all the calculated cavs. It later sorts concepts
    based on their TCAV scores.

    Args:
      test: If true, perform statistical testing and removes concepts that don't
        pass
      sort: If true, it will sort concepts in each bottleneck layers based on
        average TCAV score of the concept.
      tcav_score_images: Target class images used for calculating tcav scores.
        If None, the target class source directory images are used.

    Returns:
      A dictionary of the form {'bottleneck layer':{'concept name':
      [list of tcav scores], ...}, ...} containing TCAV scores.
    """

        tcav_scores = {bn: {} for bn in self.bottlenecks}
        randoms = ['random500_{}'.format(i) for i in np.arange(self.num_random_exp)]
        if tcav_score_videos is None:  # Load target class images if not given
            raw_videos_list = self.load_concept_videos(self.target_class, 2 * self.max_videos)
            tcav_score_videos_list = raw_videos_list[-self.max_videos:]
        gradients = self._return_gradients_single(tcav_score_videos_list)
        for bn in self.bottlenecks:
            for concept in self.dic[bn]['concepts'] + [self.random_concept]:
                def t_func(rnd):
                    return self._tcav_score(bn, concept, rnd, gradients)

                if self.num_workers:
                    pool = multiprocessing.Pool(self.num_workers)
                    tcav_scores[bn][concept] = pool.map(lambda rnd: t_func(rnd), randoms)
                else:
                    tcav_scores[bn][concept] = [t_func(rnd) for rnd in randoms]
        if test:
            self.test_and_remove_concepts(tcav_scores)
        if sort:
            self._sort_concepts(tcav_scores)

        return tcav_scores

    def do_statistical_testings(self, i_ups_concept, i_ups_random):
        """Conducts ttest to compare two set of samples.

    In particular, if the means of the two samples are staistically different.

    Args:
      i_ups_concept: samples of TCAV scores for concept vs. randoms
      i_ups_random: samples of TCAV scores for random vs. randoms

    Returns:
      p value
    """
        min_len = min(len(i_ups_concept), len(i_ups_random))
        _, p = stats.ttest_rel(i_ups_concept[:min_len], i_ups_random[:min_len])
        return p

    def test_and_remove_concepts(self, tcav_scores):
        """Performs statistical testing for all discovered concepts.

    Using TCAV socres of the discovered concepts versurs the random_counterpart
    concept, performs statistical testing and removes concepts that do not pass

    Args:
      tcav_scores: Calculated dicationary of tcav scores of all concepts
    """
        concepts_to_delete = []
        for bn in self.bottlenecks:
            for concept in self.dic[bn]['concepts']:
                pvalue = self.do_statistical_testings \
                    (tcav_scores[bn][concept], tcav_scores[bn][self.random_concept])
                if pvalue > 0.01:
                    concepts_to_delete.append((bn, concept))
        for bn, concept in concepts_to_delete:
            self.delete_concept(bn, concept)

    def delete_concept(self, bn, concept):
        """Removes a discovered concepts if it's not already removed.

    Args:
      bn: Bottleneck layer where the concepts is discovered.
      concept: concept name
    """
        self.dic[bn].pop(concept, None)
        if concept in self.dic[bn]['concepts']:
            self.dic[bn]['concepts'].pop(self.dic[bn]['concepts'].index(concept))

    def _concept_profile(self, bn, activations, concept, randoms):
        """Transforms data points from activations space to concept space.

    Calculates concept profile of data points in the desired bottleneck
    layer's activation space for one of the concepts

    Args:
      bn: Bottleneck layer
      activations: activations of the data points in the bottleneck layer
      concept: concept name
      randoms: random concepts

    Returns:
      The projection of activations of all images on all CAV directions of
        the given concept
    """

        def t_func(rnd):
            products = self.load_cav_direction(concept, rnd, bn) * activations
            return np.sum(products, -1)

        if self.num_workers:
            pool = multiprocessing.Pool(self.num_workers)
            profiles = pool.map(lambda rnd: t_func(rnd), randoms)
        else:
            profiles = [t_func(rnd) for rnd in randoms]
        return np.stack(profiles, axis=-1)

    def find_profile(self, bn, videos, mean=True):
        """Transforms images from pixel space to concept space.

    Args:
      bn: Bottleneck layer
      videos: Data points to be transformed
      mean: If true, the profile of each concept would be the average inner
        product of all that concepts' CAV vectors rather than the stacked up
        version.

    Returns:
      The concept profile of input images in the bn layer.
    """
        profile = np.zeros((len(videos), len(self.dic[bn]['concepts']),
                            self.num_random_exp))
        class_acts = get_acts_from_videos(
            videos, self.model, bn).reshape([len(videos), -1])
        randoms = ['random500_{}'.format(i) for i in range(self.num_random_exp)]
        for i, concept in enumerate(self.dic[bn]['concepts']):
            profile[:, i, :] = self._concept_profile(bn, class_acts, concept, randoms)
        if mean:
            profile = np.mean(profile, -1)
        return profile
