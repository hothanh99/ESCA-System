from core.Preprocessing import Feature_extractor
import tensorflow as tf
import os, sys
sys.path.append(os.getcwd())
from RealtimeTraining import rt_config
from pydub import AudioSegment
from pydub.utils import make_chunks
from helper.utils import read_file_name
from tqdm import tqdm
import numpy as np
from shutil import copy
from RealtimeTraining import rt_config

class Dataloader(Feature_extractor):
    def __init__(self, cfg, idx_realtime):
        super().__init__(
            type=cfg.PREPROCESS.TYPE, segment_len=cfg.PREPROCESS.SEGMENT_LEN, \
            audio_len=cfg.PREPROCESS.AUDIO_LEN, sample_per_file=cfg.PREPROCESS.SAMPLE_PER_FILE, \
            window_time=cfg.PREPROCESS.GAMMA.WINDOW_TIME, hop_time=cfg.PREPROCESS.GAMMA.HOP_TIME, \
            channels=cfg.PREPROCESS.GAMMA.CHANNELS, f_min=cfg.PREPROCESS.GAMMA.F_MIN, \
            sr=cfg.PREPROCESS.MEL.SR, nfft=cfg.PREPROCESS.MEL.NFFT, n_mel_band=cfg.PREPROCESS.MEL.N_BANDS
        )
        self.idx_realtime = idx_realtime
        # some paths to data directories
        self.src_data_dir = {
            'tempChunks':   rt_config.TEMP_DATA_REALTIME,
            'data':  rt_config.DATA,
            'tempSplittedRealtime': rt_config.TEMP_SPLITTED_REALTIME
        }
        self.test_data_dir = cfg.DATASET.PATH.TEST
        self.tfrecord_dir = {
            'train':    os.path.join(cfg.DATASET.PATH.TFRECORDS, 'train'),
            'val':      os.path.join(cfg.DATASET.PATH.TFRECORDS, 'val'),
            'test':     os.path.join(cfg.DATASET.PATH.TFRECORDS, 'test'),
            'normal_tl':  os.path.join(cfg.DATASET.PATH.TFRECORDS, 'train'),
            'anomaly_tl': os.path.join(cfg.DATASET.PATH.TFRECORDS, 'anomaly'),
        }
        self.stat_path = cfg.DATASET.PATH.TFRECORDS
        self.impl_func = {
            'wav': self._create_tfrecord_from_wav
        }['wav']
        self.anomaly_tfrecord_dir = os.path.join(cfg.DATASET.PATH.TFRECORDS, 'anomaly')
        os.makedirs(self.anomaly_tfrecord_dir, exist_ok=True)

        self.train_data_ratio = cfg.DATASET.RATIO.TRAIN
        self.test_data_ratio = cfg.DATASET.RATIO.TEST

        # parameters for creating dataloader
        self.batch_size = cfg.DATASET.DATALOADER.BATCH_SIZE
        self.shuffle = cfg.DATASET.DATALOADER.SHUFFLE


    def _check_for_test(self, file_list, file_nums):
        '''
            Check if DATASET.PATH.TEST parameter is assigned
            If not, part of normal training data will be used for testing
        '''
        train_idx = int(self.train_data_ratio*file_nums)
        test_idx = int(self.test_data_ratio*file_nums)
        if self.test_data_dir:
            print(f"Getting test data from {self.test_data_dir}")
            data_dict = {
                'train':file_list[:train_idx+test_idx],
                'normal_test': read_file_name(self.test_data_dir),
                'val': file_list[train_idx+test_idx:],
            }
        else:
            data_dict = {
                'train':file_list[:train_idx],
                'normal_test': file_list[train_idx:train_idx+test_idx],
                'val': file_list[train_idx+test_idx:],
            }

        return data_dict

    def create_tfrecord(self):
        self.impl_func()

    def _create_tfrecord_from_wav(self):
        def _bytes_feature(value):
            """Returns a bytes_list from a string / byte."""
            # first serialize the input
            value = tf.io.serialize_tensor(value)
            if isinstance(value, type(tf.constant(0))):
                value = value.numpy() # BytesList won't unpack a string from an EagerTensor.
            return tf.train.Feature(bytes_list=tf.train.BytesList(value=[value]))

        time_per_sample = self.segment_len*1000  # time is processed in millisecond
        rate = self.audio_len//self.segment_len
        # read all files in the directory
        test_id_holder = 0 # a work around to save both anomaly test sample and normal test sample to the same directory
        file = self.src_data_dir['tempChunks'] + f'\\realtime_{self.idx_realtime}.wav'
        name = file.split('\\')[-1][:-4]
        tfrecordFolderPath = rt_config.TFRECORD_REALTIME + f'\\{name}'
                
        if os.path.exists(tfrecordFolderPath):
            pass
        else: 
            os.mkdir(tfrecordFolderPath)
        sample_counter = 0
        idx_list = []
        feature_list = []
        filePaths = []
        audio = AudioSegment.from_file(file, 'wav')
        chunks = make_chunks(audio, time_per_sample)
        for idx, chunk in tqdm(enumerate(chunks)):
            feature = self.feat_extr_func[self.type](chunk)
            feature_list.append(feature)
            idx_list.append(name +'_'+str(idx))
            sample_counter += 1
            print(idx_list)
            filePath = os.path.join(tfrecordFolderPath + f'\\data_{idx:08}.tfrecord')
            filePaths.append(filePath)
        for feature, id, Path in zip(feature_list, idx_list, filePaths):
            filePath = os.path.join(tfrecordFolderPath + f'\\data_{idx:08}.tfrecord')
            with tf.io.TFRecordWriter(Path) as writer:
                print(feature_list)
                
                temp = tf.train.Example(features=tf.train.Features(
                    feature={
                        'feature':  _bytes_feature(feature.astype(np.float32)),
                        'label':    _bytes_feature(0),
                        'idx':      _bytes_feature(id),
                    }
                )).SerializeToString()
                writer.write(temp)
            
                
            
            # file_nums = len(file_list)
            # if not file_nums:
            #     print(f'{src_dir} directory is empty.')
            #     return None

            # if 'normal' in data_type:
            #     data_dict = self._check_for_test(file_list, file_nums)
            #     label = 0
            # else:
            #     data_dict = {
            #         'anomaly_test': file_list
            #     }
            #     label = 1


    def _create_tfrecord_from_npz(self):
        pass

    def _check_idx_exist(self):
        pass

    def create_dataloader(self, data_part, batch_size=None):
        feature_description = {
            'feature':  tf.io.FixedLenFeature([], tf.string),
            'label':    tf.io.FixedLenFeature([], tf.string),
            'idx':      tf.io.FixedLenFeature([], tf.string),
        }

        def _parse_function(input_proto):
            # Parse the input `tf.train.Example` proto using the dictionary feature_description.
            parsed_sample = tf.io.parse_single_example(input_proto, feature_description)
            return  (tf.io.parse_tensor(parsed_sample['feature'], tf.float32), \
                    tf.io.parse_tensor(parsed_sample['label'], tf.int32), parsed_sample['idx'])

        abs_path = lambda x: os.path.join(self.tfrecord_dir[data_part], x)
        tfrecords_list = list(map(abs_path, os.listdir(self.tfrecord_dir[data_part])))

        dataset = tf.data.TFRecordDataset(tfrecords_list)
        parsed_dataset = dataset.map(_parse_function)
        bs = batch_size if batch_size else self.batch_size
        if self.shuffle:
            parsed_dataset = parsed_dataset.shuffle(buffer_size=1000)

        return parsed_dataset.batch(batch_size=bs)

    def create_dataloader_from_files(self, list_of_files, batch_size=None):
        feature_description = {
            'feature':  tf.io.FixedLenFeature([], tf.string),
            'label':    tf.io.FixedLenFeature([], tf.string),
            'idx':      tf.io.FixedLenFeature([], tf.string),
        }

        def _parse_function(input_proto):
            # Parse the input `tf.train.Example` proto using the dictionary feature_description.
            parsed_sample = tf.io.parse_single_example(input_proto, feature_description)
            return  (tf.io.parse_tensor(parsed_sample['feature'], tf.float32), \
                    tf.io.parse_tensor(parsed_sample['label'], tf.int32), parsed_sample['idx'])

        dataset = tf.data.TFRecordDataset(list_of_files)
        parsed_dataset = dataset.map(_parse_function)
        bs = batch_size if batch_size else self.batch_size
        if self.shuffle:
            parsed_dataset = parsed_dataset.shuffle(buffer_size=1000)

        return parsed_dataset.batch(batch_size=bs)

    def accumulate_stat(self):
        train_data = self.create_dataloader('train', 1)

        MIN = np.iinfo(np.int16).max
        MAX = np.iinfo(np.int16).min

        for feature, _, _ in tqdm(train_data, desc='Accumulating statistics'):
            temp1 = tf.reduce_min(feature)
            temp2 = tf.reduce_max(feature)
            MIN = MIN if MIN<temp1 else temp1
            MAX = MAX if MAX>temp2 else temp2

        with open(os.path.join(self.stat_path, 'stats.npz'), 'wb') as file:
            np.savez(file, max=MAX, min=MIN)
