"""This script runs the whole ACE method.
this is ace for video first resize to 112*112 for segmentation and use all frames
we don't set down size in this version, videos are directly resized to (image.shape[1], image.shape[2])
"""
import sys
import os
from tcavvideo import utils
import tensorflow as tf
import ace_helpers
from ace import ConceptDiscovery
import argparse
import pickle


def main(args):
    mode = 'train'
    discovered_concepts_dir = os.path.join(args.working_dir, 'concepts/')
    # save patches and supervoxels for each video considering memory
    patches_dir = os.path.join(args.working_dir, 'patches')
    supervoxels_dir = os.path.join(args.working_dir, 'supervoxels')
    masks_dir = os.path.join(args.working_dir, 'masks')
    results_dir = os.path.join(args.working_dir, 'results/')
    cavs_dir = os.path.join(args.working_dir, 'cavs/')
    activations_dir = os.path.join(args.working_dir, 'acts/')
    results_summaries_dir = os.path.join(args.working_dir, 'results_summaries/')
    if tf.io.gfile.exists(patches_dir) and os.listdir(patches_dir):
        print('Continue to process...')
    else:
        tf.io.gfile.makedirs(args.working_dir)
        tf.io.gfile.makedirs(discovered_concepts_dir)
        tf.io.gfile.makedirs(results_dir)
        tf.io.gfile.makedirs(cavs_dir)
        tf.io.gfile.makedirs(activations_dir)
        tf.io.gfile.makedirs(masks_dir)
        tf.io.gfile.makedirs(results_summaries_dir)
        tf.io.gfile.makedirs(patches_dir)
        tf.io.gfile.makedirs(supervoxels_dir)

    random_concept = 'random_discovery'  # Random concept for statistical testing
    sess = utils.create_session()
    mymodel = ace_helpers.make_model(
        sess, args.model_to_run, args.model_path, args.labels_path, args.imageshape)
    cd = ConceptDiscovery(
        mymodel,
        args.target_class,
        random_concept,
        args.bottlenecks.split(','),
        sess,
        args.working_dir,
        discovered_concepts_dir,
        args.source_dir,
        activations_dir,
        cavs_dir,
        patches_dir,
        supervoxels_dir,
        masks_dir,
        num_random_exp=args.num_random_exp,
        channel_mean=True,
        max_videos=args.max_videos,
        min_videos=args.min_videos,
        num_discovery_videos=args.max_videos,
        num_workers=args.num_parallel_workers,
        imageshape=args.imageshape,
        mode=mode,
        bs=args.batchsize)
    # Creating the dataset of video patches
    cd.create_patches(param_dict={'n_segments': [15, 50, 80]})
    # Discovering Concepts, use kmeans to cluster voxels into 25 classes
    cd.discover_concepts(method='KM', param_dicts={'n_clusters': 25})
    del cd.dataset  # Free memory
    del cd.video_numbers
    del cd.patches
    # Save discovered concept images (resized and original sized)
    ace_helpers.save_concepts(cd, discovered_concepts_dir)
    # Calculating CAVs and TCAV scores
    # here return all concepts/target class/random data with random data classification accuracies
    cav_accuracies = cd.cavs(min_acc=0.0)
    scores = cd.tcavs(test=False)
    tcav = ace_helpers.save_ace_report(cd, cav_accuracies, scores,
                                       results_summaries_dir + 'ace_results.txt')
    # Plot examples of discovered concepts
    for bn in cd.bottlenecks:
        ace_helpers.plot_concepts(cd, bn, 10, address=results_dir)
    # Delete concepts that don't pass statistical testing
    train_data = [tcav, cd.dic]
    f = open(args.working_dir + '/' + 'train_data.pkl', "wb")
    pickle.dump(train_data, f)
    f.close()

    cd.test_and_remove_concepts(scores)


def parse_arguments(argv):
    """Parses the arguments passed to the run.py script."""
    parser = argparse.ArgumentParser()
    parser.add_argument('--source_dir', type=str,
                        help='''Directory where the network's classes video folders and random
      concept folders are saved.''', default='./ImageNet')
    parser.add_argument('--working_dir', type=str,
                        help='Directory to save the results.', default='./ACE')
    parser.add_argument('--model_to_run', type=str,
                        help='The name of the model.', default='3Dcnn')
    parser.add_argument('--model_path', type=str,
                        help='Path to model checkpoints.', default='./tensorflow_inception_graph.pb')
    parser.add_argument('--labels_path', type=str,
                        help='Path to model checkpoints.', default='./imagenet_labels.txt')
    parser.add_argument('--target_class', type=str,
                        help='The name of the target class to be interpreted', default='zebra')
    parser.add_argument('--bottlenecks', type=str,
                        help='Names of the target layers of the network (comma separated)',
                        default='dense_1')
    parser.add_argument('--num_random_exp', type=int,
                        help="Number of random experiments used for statistical testing, etc",
                        default=20)
    parser.add_argument('--max_videos', type=int,
                        help="Maximum number of videos in a discovered concept",
                        default=20)
    parser.add_argument('--min_videos', type=int,
                        help="Minimum number of videos in a discovered concept",
                        default=10)
    parser.add_argument('--num_parallel_workers', type=int,
                        help="Number of parallel jobs.",
                        default=0)
    parser.add_argument('--imageshape', '--list', nargs='+',
                        help="Input image shape for 3Dnetwork, input as rows columns and depths")
    parser.add_argument('--downsize', nargs='+', type=int,
                        help="....")
    parser.add_argument('--batchsize', type=int,
                        help="batchsize for calculating activation")
    return parser.parse_args(argv)


if __name__ == '__main__':
    main(parse_arguments(sys.argv[1:]))
