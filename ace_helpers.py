""" collection of various helper functions for running ACE"""

from multiprocessing import dummy as multiprocessing
import sys
import os
from matplotlib import pyplot as plt
import matplotlib.gridspec as gridspec
import tcavvideo.model as model
import numpy as np
from PIL import Image
from skimage.segmentation import mark_boundaries
from sklearn import linear_model
from sklearn.model_selection import cross_val_score
import tensorflow as tf
import cv2
import skvideo.io
import math
import pickle


def make_model(sess, model_to_run, model_path,
               labels_path, imageshape, randomize=False, ):
    """Make an instance of a model.

  Args:
    sess: tf session instance.
    model_to_run: a string that describes which model to make.
    model_path: Path to models saved graph.
    imageshape: shape of input video size
    randomize: Start with random weights
    labels_path: Path to models line separated class names text file.

  Returns:
    a model instance.

  Raises:
    ValueError: If model name is not valid.
  """
    if model_to_run == 'InceptionV3':
        mymodel = model.InceptionV3Wrapper_public(
            sess, model_saved_path=model_path, labels_path=labels_path)
    elif model_to_run == 'GoogleNet':
        # common_typos_disable
        mymodel = model.GoogleNetWrapper_public(
            sess, model_saved_path=model_path, labels_path=labels_path)
    elif 'keras' in model_to_run:
        # mymodel = model.KerasModelWrapper(
        #     sess, model_path, labels_path, imageshape)
        mymodel = model.KerasModelWrapper(
            sess, model_path, labels_path)
    else:
        raise ValueError('Invalid model name')
    if randomize:  # randomize the network!
        sess.run(tf.compat.v1.global_variables_initializer())
    return mymodel


def load_video_from_file(filename, image_shape):
    """Given a filename without resized to small size or downsampling, try to open the file. If failed, return None.
Args:
  filename: location of the video file. if for segment, then it don't need to be resized
  image_shape: resize videos to [112, 112]
Returns:
  the video if succeeds, None if fails.
Rasies:
  exception if the image was not the right shape.
"""
    if not tf.io.gfile.exists(filename):
        tf.compat.v1.logging.error('Cannot find file: {}'.format(filename))
        return None
    try:
        cap = cv2.VideoCapture(filename)
        video = []
        ret = True
        while ret:
            ret, frame = cap.read()  # read one frame from the 'capture' object; img is (H, W, C)
            if ret:
                frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
                frame = cv2.resize(frame, (image_shape[1], image_shape[2]))
                video.append(frame)
        video = np.array(video)
        if not (len(video.shape) == 4 and video.shape[3] == 3):
            return None
        else:
            return video

    except Exception as e:
        tf.compat.v1.logging.info(e)
        return None
    return video


def get_video_list(filenames, max_videos=500, do_shuffle=True):
    # First shuffle a copy of the filenames.
    filenames = filenames[:]
    if do_shuffle:
        np.random.shuffle(filenames)
    return filenames[0:max_videos]


def get_acts_from_videos(videos, model, bottleneck_name):
    """Run images in the model to get the activations.
  Args:
    videos: a list of images
    model: a model instance
    bottleneck_name: bottleneck name to get the activation from
  Returns:
    numpy array of activations.
  """
    return np.asarray(model.run_examples(videos, bottleneck_name))


def flat_profile(cd, images, bottlenecks=None):
    """Returns concept profile of given images.

  Given a ConceptDiscovery class instance and a set of images, and desired
  bottleneck layers, calculates the profile of each image with all concepts and
  returns a profile vector

  Args:
    cd: The concept discovery class instance
    images: The images for which the concept profile is calculated
    bottlenecks: Bottleck layers where the profile is calculated. If None, cd
      bottlenecks will be used.

  Returns:
    The concepts profile of input images using discovered concepts in
    all bottleneck layers.

  Raises:
    ValueError: If bottlenecks is not in right format.
  """
    profiles = []
    if bottlenecks is None:
        bottlenecks = list(cd.dic.keys())
    if isinstance(bottlenecks, str):
        bottlenecks = [bottlenecks]
    elif not isinstance(bottlenecks, list) and not isinstance(bottlenecks, tuple):
        raise ValueError('Invalid bottlenecks parameter!')
    for bn in bottlenecks:
        profiles.append(cd.find_profile(str(bn), images).reshape((len(images), -1)))
    profile = np.concatenate(profiles, -1)
    return profile


def cross_val(a, b, methods):
    """Performs cross validation for a binary classification task.

  Args:
    a: First class data points as rows
    b: Second class data points as rows
    methods: The sklearn classification models to perform cross-validation on

  Returns:
    The best performing trained binary classification odel
  """
    x, y = binary_dataset(a, b)
    best_acc = 0.
    if isinstance(methods, str):
        methods = [methods]
    best_acc = 0.
    for method in methods:
        temp_acc = 0.
        params = [10 ** e for e in [-4, -3, -2, -1, 0, 1, 2, 3]]
        for param in params:
            clf = give_classifier(method, param)
            acc = cross_val_score(clf, x, y, cv=min(100, max(2, int(len(y) / 10))))
            if np.mean(acc) > temp_acc:
                temp_acc = np.mean(acc)
                best_param = param
        if temp_acc > best_acc:
            best_acc = temp_acc
            final_clf = give_classifier(method, best_param)
    final_clf.fit(x, y)
    return final_clf, best_acc


def give_classifier(method, param):
    """Returns an sklearn classification model.

  Args:
    method: Name of the sklearn classification model
    param: Hyperparameters of the sklearn model

  Returns:
    An untrained sklearn classification model

  Raises:
    ValueError: if the model name is invalid.
  """
    if method == 'logistic':
        return linear_model.LogisticRegression(C=param)
    elif method == 'sgd':
        return linear_model.SGDClassifier(alpha=param)
    else:
        raise ValueError('Invalid model!')


def binary_dataset(pos, neg, balanced=True):
    """Creates a binary dataset given instances of two classes.

  Args:
     pos: Data points of the first class as rows
     neg: Data points of the second class as rows
     balanced: If true, it creates a balanced binary dataset.

  Returns:
    The data points of the created data set as rows and the corresponding labels
  """
    if balanced:
        min_len = min(neg.shape[0], pos.shape[0])
        ridxs = np.random.permutation(np.arange(2 * min_len))
        x = np.concatenate([neg[:min_len], pos[:min_len]], 0)[ridxs]
        y = np.concatenate([np.zeros(min_len), np.ones(min_len)], 0)[ridxs]
    else:
        ridxs = np.random.permutation(np.arange(len(neg) + len(pos)))
        x = np.concatenate([neg, pos], 0)[ridxs]
        y = np.concatenate(
            [np.zeros(neg.shape[0]), np.ones(pos.shape[0])], 0)[ridxs]
    return x, y


def plot_concepts(cd, bn, num=10, address=None, mode='diverse', concepts=None):
    """Plots examples of discovered concepts.

  Args:
    cd: The concept discovery instance
    bn: Bottleneck layer name
    num: Number of images to print out of each concept, for video, the middle frames are plotted
    address: If not None, saves the output to the address as a .PNG image
    mode: If 'diverse', it prints one example of each of the target class images
      is coming from. If 'radnom', randomly samples exmples of the concept. If
      'max', prints out the most activating examples of that concept.
    concepts: If None, prints out examples of all discovered concepts.
      Otherwise, it should be either a list of concepts to print out examples of
      or just one concept's name

  Raises:
    ValueError: If the mode is invalid.
  """
    if concepts is None:
        concepts = cd.dic[bn]['concepts']
    elif not isinstance(concepts, list) and not isinstance(concepts, tuple):
        concepts = [concepts]
    num_concepts = len(concepts)
    plt.rcParams['figure.figsize'] = num * 2.1, 4.3 * num_concepts
    fig = plt.figure(figsize=(num * 2, 4 * num_concepts))
    outer = gridspec.GridSpec(num_concepts, 1, wspace=0., hspace=0.3)
    for n, concept in enumerate(concepts):
        inner = gridspec.GridSpecFromSubplotSpec(
            2, num, subplot_spec=outer[n], wspace=0, hspace=0.1)
        concept_videos = []
        concept_patches = []
        for patch_path in cd.dic[bn][concept]['patches']:
            concept_patches.append(pickle.load(open(patch_path, 'rb')))
        for video_path in cd.dic[bn][concept]['videos']:
            concept_videos.append(pickle.load(open(video_path, 'rb')))
        concept_video_numbers = cd.dic[bn][concept]['video_numbers']
        concept_index = cd.dic[bn][concept]['concept_index']
        if mode == 'max':
            idxs = np.arange(len(concept_videos))
        elif mode == 'random':
            idxs = np.random.permutation(np.arange(len(concept_videos)))
        elif mode == 'diverse':
            idxs = []
            while True:
                seen = set()
                for idx in range(len(concept_videos)):
                    if concept_video_numbers[idx] not in seen and idx not in idxs:
                        seen.add(concept_video_numbers[idx])
                        idxs.append(idx)
                if len(idxs) == len(concept_videos):
                    break
        else:
            raise ValueError('Invalid mode!')
        idxs = idxs[:num]
        for i, idx in enumerate(idxs):
            ax = plt.Subplot(fig, inner[i])
            ax.imshow(concept_videos[idx][math.trunc(len(concept_videos[idx])/2)].astype(np.uint8))
            ax.set_xticks([])
            ax.set_yticks([])
            if i == int(num / 2):
                ax.set_title(concept)
            ax.grid(False)
            fig.add_subplot(ax)
            ax = plt.Subplot(fig, inner[i + num])
            mask = 1 - (np.mean(concept_patches[idx][math.trunc(len(concept_patches[idx])/2)] == float(
                cd.average_image_value), -1) == 1)
            video_dir = cd.discovery_videos[concept_video_numbers[idx]]
            video = load_video_from_file(video_dir, cd.image_shape).astype(np.uint8)
            start, end = concept_index[idx]
            image = video[start + math.trunc((end-start+1)/2)]
            ax.imshow(mark_boundaries(image, mask, color=(1, 1, 0), mode='thick'))
            ax.set_xticks([])
            ax.set_yticks([])
            ax.set_title(str(concept_video_numbers[idx]))
            ax.grid(False)
            fig.add_subplot(ax)
            del video
    plt.suptitle(bn)
    if address is not None:
        with tf.io.gfile.GFile(address + bn + '.png', 'w') as f:
            fig.savefig(f)
        plt.clf()
        plt.close(fig)


def cosine_similarity(a, b):
    """Cosine similarity of two vectors."""
    assert a.shape == b.shape, 'Two vectors must have the same dimensionality'
    a_norm, b_norm = np.linalg.norm(a), np.linalg.norm(b)
    if a_norm * b_norm == 0:
        return 0.
    cos_sim = np.sum(a * b) / (a_norm * b_norm)
    return cos_sim


def similarity(cd, num_random_exp=None, num_workers=25):
    """Returns cosine similarity of all discovered concepts.

  Args:
    cd: The ConceptDiscovery module for discovered conceps.
    num_random_exp: If None, calculates average similarity using all the class's
      random concepts. If a number, uses that many random counterparts.
    num_workers: If greater than 0, runs the function in parallel.

  Returns:
    A similarity dict in the form of {(concept1, concept2):[list of cosine
    similarities]}
  """

    def concepts_similarity(cd, concepts, rnd, bn):
        """Calcualtes the cosine similarity of concept cavs.

    This function calculates the pairwise cosine similarity of all concept cavs
    versus an specific random concept

    Args:
      cd: The ConceptDiscovery instance
      concepts: List of concepts to calculate similarity for
      rnd: a random counterpart
      bn: bottleneck layer the concepts belong to

    Returns:
      A dictionary of cosine similarities in the form of
      {(concept1, concept2): [list of cosine similarities], ...}
    """
        similarity_dic = {}
        for c1 in concepts:
            cav1 = cd.load_cav_direction(c1, rnd, bn)
            for c2 in concepts:
                if (c1, c2) in similarity_dic.keys():
                    continue
                cav2 = cd.load_cav_direction(c2, rnd, bn)
                similarity_dic[(c1, c2)] = cosine_similarity(cav1, cav2)
                similarity_dic[(c2, c1)] = similarity_dic[(c1, c2)]
        return similarity_dic

    similarity_dic = {bn: {} for bn in cd.bottlenecks}
    if num_random_exp is None:
        num_random_exp = cd.num_random_exp
    randoms = ['random500_{}'.format(i) for i in np.arange(num_random_exp)]
    concepts = {}
    for bn in cd.bottlenecks:
        concepts[bn] = [cd.target_class, cd.random_concept] + cd.dic[bn]['concepts']
    for bn in cd.bottlenecks:
        concept_pairs = [(c1, c2) for c1 in concepts[bn] for c2 in concepts[bn]]
        similarity_dic[bn] = {pair: [] for pair in concept_pairs}

        def t_func(rnd):
            return concepts_similarity(cd, concepts[bn], rnd, bn)

        if num_workers:
            pool = multiprocessing.Pool(num_workers)
            sims = pool.map(lambda rnd: t_func(rnd), randoms)
        else:
            sims = [t_func(rnd) for rnd in randoms]
        while sims:
            sim = sims.pop()
            for pair in concept_pairs:
                similarity_dic[bn][pair].append(sim[pair])
    return similarity_dic


def save_ace_report(cd, accs, scores, address):
    """Saves TCAV scores.

  Saves the average CAV accuracies and average TCAV scores of the concepts
  discovered in ConceptDiscovery instance.

  Args:
    cd: The ConceptDiscovery instance.
    accs: The cav accuracy dictionary returned by cavs method of the
      ConceptDiscovery instance
    scores: The tcav score dictionary returned by tcavs method of the
      ConceptDiscovery instance
    address: The address to save the text file in.
  """
    report = '\n\n\t\t\t ---CAV accuracies---'
    for bn in cd.bottlenecks:
        report += '\n'
        for concept in cd.dic[bn]['concepts']:
            report += '\n' + bn + ':' + concept + ':' + str(
                np.mean(accs[bn][concept]))

    # with tf.io.gfile.GFile(address, 'w') as f:
    #     f.write(report)
    # report = '\n\n\t\t\t ---TCAV scores---'
    report += '\n\n\t\t\t ---TCAV scores---'
    tcav = {bn: {} for bn in cd.bottlenecks}
    for bn in cd.bottlenecks:
        report += '\n'
        for concept in cd.dic[bn]['concepts']:
            pvalue = cd.do_statistical_testings(
                scores[bn][concept], scores[bn][cd.random_concept])
            tcav[bn][concept] = [np.mean(scores[bn][concept]), pvalue]
            report += '\n{}:{}:{},{}'.format(bn, concept,
                                             np.mean(scores[bn][concept]), pvalue)
    with tf.io.gfile.GFile(address, 'w') as f:
        f.write(report)
    return tcav


def save_concepts(cd, concepts_dir):
    """Saves discovered concept's images or patches.

  Args:
    cd: The ConceptDiscovery instance the concepts of which we want to save
    concepts_dir: The directory to save the concept images
  """
    for bn in cd.bottlenecks:
        for concept in cd.dic[bn]['concepts']:
            patches_dir = os.path.join(concepts_dir, bn + '_' + concept + '_patches')
            videos_dir = os.path.join(concepts_dir, bn + '_' + concept)
            patch_video = []
            vvideo = []
            for patch_path in cd.dic[bn][concept]['patches']:
                patch_video.append(pickle.load(open(patch_path, 'rb')))
            for video_path in cd.dic[bn][concept]['videos']:
                vvideo.append(pickle.load(open(video_path, 'rb')))
            # patches = [(np.clip(x, 0, 1) * 255).astype(np.uint8) for x in patch_video]
            # videos = [(np.clip(x, 0, 1) * 255).astype(np.uint8) for x in vvideo]
            patches = [x.astype(np.uint8) for x in patch_video]
            videos = [x.astype(np.uint8) for x in vvideo]
            tf.io.gfile.makedirs(patches_dir)
            tf.io.gfile.makedirs(videos_dir)
            video_numbers = cd.dic[bn][concept]['video_numbers']
            video_addresses, patch_addresses = [], []
            for i in range(len(videos)):
                video_name = '0' * int(np.ceil(2 - np.log10(i + 1))) + '{}_{}'.format(
                    i + 1, video_numbers[i])
                patch_addresses.append(os.path.join(patches_dir, video_name + '.avi'))
                video_addresses.append(os.path.join(videos_dir, video_name + '.avi'))
            save_videos(patch_addresses, patches)
            save_videos(video_addresses, videos)


def save_videos(addresses, videos):
    """Save videos in the addresses.

  Args:
      i: index of video
    addresses: The list of addresses to save the videos as or the address of the
      directory to save all images in. (list or str)
    videos: The list of all videos in numpy uint8 format.
  """
    if not isinstance(addresses, list):
        video_addresses = []
        for i, video in enumerate(videos):
            video_name = '0' * (3 - int(np.log10(i + 1))) + str(i + 1) + '.avi'
            video_addresses.append(os.path.join(addresses, video_name))
        addresses = video_addresses
    assert len(addresses) == len(videos), 'Invalid number of addresses'  # assert statement when false
    for address, video in zip(addresses, videos):
        height, width = video.shape[1], video.shape[2]
        size = (width, height)
        outvideo = cv2.VideoWriter(address, cv2.VideoWriter_fourcc(*'DIVX'), 25, size)
        for i in range(len(video)):
            outvideo.write(video[i])
        outvideo.release()


def save_video(i, address, video):
    """Save single video in the addresses.

  Args:
    i: index of video
    address: The list of addresses to save the videos as or the address of the
      directory to save all images in. (list or str)
    video: The list of all videos in numpy uint8 format.
  """
    video_name = '0' * (3 - int(np.log10(i + 1))) + str(i + 1) + '.avi'
    video_address = os.path.join(address, video_name)
    height, width = video.shape[1], video.shape[2]
    size = (width, height)
    outvideo = cv2.VideoWriter(video_address, cv2.VideoWriter_fourcc(*'DIVX'), 25, size)
    for i in range(len(video)):
        outvideo.write(video[i])
    outvideo.release()

