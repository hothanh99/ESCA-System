from os import listdir, scandir, rename, environ, remove, setpgrp, killpg
import os
environ['TF_CPP_MIN_LOG_LEVEL'] = '3'
from tensorflow.keras.models import load_model
from os.path import join, isdir, dirname
import time
import signal
from scipy.io import wavfile
from gammatone import gtgram
import numpy as np
import json

import csv
from datetime import datetime
import subprocess


def testing(env=None, target=None, transfer_learning=True, manual_threshold=None, rtime=300, eval=None):
    # gate keeping check
    if not env:
        raise ValueError('Please specify a enviromnet for evaluating.')
    if not target:
        raise ValueError('Please specify a target for evaluating.')
    if not env in ['intersection', 'park']:
        raise ValueError(f'The {env} enviroment has not been implemented.')

    root = dirname(__file__)

    # load threshold and model from file
    metric_path = 'Results/Updated_Graphs/vq_vae'
    model_path = 'Results/Saved_models/vq_vae'
    if target != 'source':
        type = 'tl' if transfer_learning else 'base'
        sub_path = 'target/' + env
        name = target + '_2s_32_bandx32_frame'
        metric_file = join(root, metric_path, type, sub_path, name, 'metrics_detail.json')
        model = load_model(join(root, model_path, type, sub_path, name))
    else:
        type = 'source'
        name = env + '_2s_32_bandx32_frame'
        metric_file = join(root, metric_path, type, name, 'metrics_detail.json')
        model = load_model(join(root, model_path, type, name))

    with open(metric_file, 'r') as f:
        metric = json.load(f)

    auto_th = metric['threshold']
    MAX = metric['max']
    MIN = metric['min']

    threshold = auto_th if not manual_threshold else float(manual_threshold)
    print(f'Threshold: {threshold}')


    # second load sample files
    sample_loc = join(root,'test_samples/test') # change this to the recorded file location

    # some characteristics of gammatone feature
    window_time = 0.06*2
    channels = 32
    hop_time = window_time/2
    f_min = 100
    frame_rate = 44100

    start = time.time()
    i = 0

    print(f'Real-time detection start... env:{env}, model:{type}, target:{target}')

    # a dict to store some info
    data = {
        'name': None,
        'pred': None,
        'time': None,
    }

    # prepare cvs file to log in the information
    csv_file = join(root,'test_samples', 'temp.csv')
    field_names = list(data.keys())
    with open(csv_file, 'w') as file:
        csv_writer = csv.DictWriter(file, fieldnames=field_names)
        csv_writer.writeheader()

    # run another subprocess to read from the csv file and draw graph dynamically
    plotting_graph = join(root, 'plotting_graph.py')
    command = ['python3', plotting_graph, '-th', str(threshold), '-csv', csv_file]
    graph = subprocess.Popen(command, preexec_fn=setpgrp)

    try:
        while(True):
            # load and process the audio
            base_file = listdir(sample_loc)
            end = time.time()
            if (end-start) > rtime:
                break
            if not 'basefile.wav' in base_file:
                continue

            s = join(sample_loc, 'basefile.wav')
            _, audio = wavfile.read(s)
            gtg = gtgram.gtgram(audio, frame_rate, window_time, hop_time, channels, f_min)    # noqa: E501
            a = np.flipud(20 * np.log10(gtg))
            # rescale
            a = np.clip((a-MIN)/(MAX-MIN), a_min=0, a_max=1)
            a = np.reshape(a, (1 ,a.shape[0], a.shape[1], 1))
            pred = np.mean((a-model.predict(a))**2)
            type = 1 if pred > threshold else 0

            data[field_names[0]] = i
            data[field_names[1]] = pred
            data[field_names[2]] = datetime.now().strftime("%Y%m%d-%H%M%S")

            with open(csv_file, 'a') as file:
                csv_writer = csv.DictWriter(file, fieldnames=field_names)
                csv_writer.writerow(data)

            rename(s, join(root, 'test_samples/temp', f'basefile_{i}.wav')) # move file
            i += 1
            if type == 1:
                print(f'Detect abnormal at {end - start}s from starting time.')
            else:
                print('Everything is normal.')

        print('inferencing end.')
        # wait to input any key
        var = input("Please input any key.")
        killpg(graph.pid, signal.SIGINT)
    except KeyboardInterrupt:

        print('inferencing end.')
        # wait to input any key
        var = input("Please input any key.")
        killpg(graph.pid, signal.SIGINT)

    return 0
