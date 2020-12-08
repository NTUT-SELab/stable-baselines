import os
import tensorflow as tf
import glob
import bz2
import pickle

from pathlib import Path
from datetime import datetime

class Storage:
    def __init__(self, limit=10000):
        self.space = dict()
        self.size = 0
        self.limit = limit
        self.current_step = -1

    def store(self, category, step, data):
        if category not in self.space:
            self.space[category] = dict()
        
        self.space[category][step] = data
        if self.current_step != int(step):
            self.size += 1
            self.current_step = int(step)

    def restore(self, category, step):
        if self.current_step != int(step):
            self.size -= 1
            self.current_step = step
        return self.space[category][step]

    def is_full(self):
        return self.size >= self.limit

    def is_empty(self):
        return self.size <= 0

    def reset(self):
        del self.space
        self.space = dict()
        self.size = 0

class CustomSaver:
    def __init__(self, ckpt_path, info_path, name, restore_id=None, mode=None):
        """
        Initialize Sustom Saver
        :param ckpt_path: (str) path of checkpoints
        :param info_path: (str) path of info
        :param name: (str) name of checkpoints/info
        :param name: (str) id of checkpoints/info
        :param mode: (int) None:do nothing, 0:restore, 1:save  
        """
        self.ckpt_path = ckpt_path
        self.info_path = info_path
        self.name = name
        self.mode = mode
        self.restore_id = restore_id
        self.timestamp = str(int(datetime.now().timestamp()))

    def build(self):
        self.saver = tf.train.Saver(max_to_keep=None)
        if self.mode == 0:
            if self.restore_id is not None:
                self.timestamp = str(self.restore_id)
            else:
                self.timestamp = str(self._get_latest_id(self.ckpt_path, self.name+'_checkpoints'))

        self.ckpt_path = os.path.join(self.ckpt_path, "{}_{}".format(self.name+'_checkpoints', self.timestamp))
        self.info_path = os.path.join(self.info_path, "{}_{}".format(self.name+'_info', self.timestamp))
        self._create_dirs()

    def save_or_restore_ckpt(self, sess, step):
        if self.mode == 1:
            self.saver.save(sess, os.path.join(self.ckpt_path, 'ckpt'), global_step=step)
        elif self.mode == 0:
            try:
                self.saver.restore(sess, os.path.join(self.ckpt_path, 'ckpt-' + str(step)))
            except:
                print (f'fail to restore ckpt-{step}')

    def save_storage_to_file(self, storage, last_batch_timesteps):
        if self.mode == 1 and (storage.is_full() or storage.current_step > last_batch_timesteps):
            name = self.timestamp + '_' + str(storage.current_step)+".bz2"
            with bz2.open(os.path.join(self.info_path, name), "wb") as f:
                f.write(pickle.dumps(storage))
            storage.reset()
        return storage

    def restore_storage_from_file(self, storage):
        if self.mode == 0 and storage.is_empty():
            name = self._get_storage_name(self.info_path, storage.current_step)
            with bz2.open(os.path.join(self.info_path, name), "rb") as f:
                # Decompress data from file
                content = f.read()
                storage = pickle.loads(content)
        return storage

    def _create_dirs(self):
        if self.mode == 1:
            Path(self.ckpt_path).mkdir(parents=True, exist_ok=False)
            Path(self.info_path).mkdir(parents=True, exist_ok=False)

        
    def _get_latest_id(self, save_path, name):
        """
        returns the latest run number for the given log name and log path,
        by finding the greatest number in the directories.

        :return: (int) latest run number
        """
        max_id = 0
        for path in glob.glob("{}/{}_[0-9]*".format(save_path, name)):
            file_name = path.split(os.sep)[-1]
            ext = file_name.split("_")[-1]
            if name == "_".join(file_name.split("_")[:-1]) and ext.isdigit() and int(ext) > max_id:
                max_id = int(ext)
        return max_id

    def _get_storage_name(self, save_path, step):
        min_dist = float('inf')
        for path in glob.glob("{}/{}_[0-9]*".format(save_path, self.timestamp)):
            file_name = path.split(os.sep)[-1]
            ext = file_name.split("_")[-1]
            ext = ext.split('.')[0]        
            dist = int(ext) - int(step)
            if self.timestamp == "_".join(file_name.split("_")[:-1]) and ext.isdigit() and dist > 0 and dist < min_dist:
                min_dist = dist
                storage_name = file_name
        return storage_name


        
