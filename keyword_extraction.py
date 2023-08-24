from data.process_sequence import generate_data_sequence
import os
from opts import get_opts
from pathlib import Path
import logging
import pickle
import nltk
from nltk.corpus import stopwords
from nltk.util import ngrams
from nltk.tokenize import word_tokenize
from nltk.probability import FreqDist
from nltk.tokenize import RegexpTokenizer


file_path = 'description.txt'
if os.path.exists(file_path):
    nltk.download('punkt')
    nltk.download('stopwords')

    # Load your text file
    with open('description.txt', 'r') as file:
        text = file.read()

    # Tokenize words
    tokenizer = RegexpTokenizer(r'\w+')
    words = tokenizer.tokenize(text.lower())

    # Remove stopwords
    stop_words = set(stopwords.words('english'))

    custom_stop_words = set([
    'car', 'road', 'person', 'pedestrian', 'driver', 'pedestrians', 'traffic', 'vehicle',
    '1', '2', '3', '4', '5', '10', '15', '20', '30',
    'like', 'since', 'see', 'already', 'let', 'city', 'cloudy',
    'downtown', 'almost', 'green', 'red', 'yellow', 'look', 
    'normally', 'several', 'directly', 'wide', 'large', 'group',
    'likely', 'location', 'lights', 'clear', 'college'
    ])

    final_stop_words = stop_words.union(custom_stop_words)

    filtered_words = [word for word in words if word not in final_stop_words]
    # filtered_words = [word for word in words]

    # Calculate term frequency
    bigrams = list(ngrams(filtered_words, 1))
    fdist = FreqDist(bigrams)

    # Get the most common words (keywords)
    num_keywords = 150  # Adjust as needed
    keywords = fdist.most_common(num_keywords)

    print("Generated Keywords:")
    for keyword, frequency in keywords:
        print(f"{keyword}: {frequency}")
else:
    args = get_opts()
    args.dataset_root_path = str(Path(__file__).parents[1]) + '/dataset'
    args.database_path = str(Path(__file__).parents[0]) + '/database'
    # Dataset
    args.dataset = 'PSI2.0'
    if args.dataset == 'PSI2.0':
        args.video_splits = os.path.join(args.dataset_root_path, 'PSI2.0_TrainVal/splits/PSI2_split.json')
    elif args.dataset == 'PSI1.0':
        args.video_splits = os.path.join(args.dataset_root_path, 'PSI1.0/splits/PSI1_split.json')
    else:
        raise Exception("Unknown dataset name!")

    # Task
    args.task_name = 'ped_intent'

    if args.task_name == 'ped_intent':
        args.database_file = 'intent_database_train.pkl'
        args.intent_model = True

    # intent prediction
    args.intent_num = 2  # 3 for 'major' vote; 2 for mean intent
    args.intent_type = 'mean'  # >= 0.5 --> 1 (cross); < 0.5 --> 0 (not cross)
    args.intent_loss = ['bce']  # Binary Cross-Entropy
    args.intent_disagreement = 1  # -1: not use disagreement 1: use disagreement to reweigh samples
    args.intent_positive_weight = 0.5  # Reweigh BCE loss of 0/1, 0.5 = count(-1) / count(1)

    # trajectory
    if args.task_name == 'ped_traj':
        args.database_file = 'traj_database_train.pkl'
        args.intent_model = False  # if (or not) use intent prediction module to support trajectory prediction
        args.traj_model = True
        args.traj_loss = ['bbox_l1']

    args.seq_overlap_rate = 0.9  # overlap rate for trian/val set
    args.test_seq_overlap_rate = 1  # overlap for test set. if == 1, means overlap is one frame, following PIE
    args.observe_length = 15
    if args.task_name == 'ped_intent':
        args.predict_length = 1  # only make one intent prediction
    elif args.task_name == 'ped_traj':
        args.predict_length = 45

    args.max_track_size = args.observe_length + args.predict_length
    args.crop_mode = 'enlarge'
    args.normalize_bbox = None
    # 'subtract_first_frame' #here use None, so the traj bboxes output loss is based on origianl coordinates
    # [None (paper results) | center | L2 | subtract_first_frame (good for evidential) | divide_image_size]

    # Model
    args.model_name = 'lstm_int_bbox'  # LSTM module, with bboxes sequence as input, to predict intent
    args.load_image = True  # only bbox sequence as input
    if args.load_image:
        logging.info('Loading Imgs to Backbone')

    else:
        args.backbone = None
        args.freeze_backbone = False

    # Train
    args.epochs = 1
    args.batch_size = 128
    args.lr = 1e-3
    args.loss_weights = {
        'loss_intent': 1.0,
        'loss_traj': 0.0,
        'loss_driving': 0.0
    }
    args.val_freq = 1
    args.test_freq = 1
    args.print_freq = 10

    with open(os.path.join(args.database_path, 'intent_database_train.pkl'), 'rb') as fid:
        imdb_train = pickle.load(fid)
    train_seq = generate_data_sequence('train', imdb_train, args)

    with open(os.path.join(args.database_path, 'intent_database_val.pkl'), 'rb') as fid:
        imdb_val = pickle.load(fid)
    val_seq = generate_data_sequence('val', imdb_val, args)

    written_sentences = set()

    with open(file_path, 'w') as f:
        for item in train_seq['description']:
            if isinstance(item, list):
                for item2 in item:
                    if isinstance(item2, list):
                        for item3 in item2:
                            if item3 not in written_sentences:
                                f.write(item3 + '\n')
                                written_sentences.add(item3)
        for item in val_seq['description']:
            if isinstance(item, list):
                for item2 in item:
                    if isinstance(item2, list):
                        for item3 in item2:
                            if item3 not in written_sentences:
                                f.write(item3 + '\n')
                                written_sentences.add(item3)

    # with open(file_path, 'w') as f:
    #     for item in train_seq['description']:
    #         if isinstance(item, list):
    #             for item2 in item:
    #                 if isinstance(item2, list):
    #                     for item3 in item2:
    #                         item3_hash = hash(item3)
    #                         if item3_hash not in written_sentences:
    #                             f.write(item3 + '\n')
    #                             written_sentences.add(item3_hash)
    #     for item in val_seq['description']:
    #         if isinstance(item, list):
    #             for item2 in item:
    #                 if isinstance(item2, list):
    #                     for item3 in item2:
    #                         item3_hash = hash(item3)
    #                         if item3_hash not in written_sentences:
    #                             f.write(item3 + '\n')
    #                             written_sentences.add(item3_hash)



