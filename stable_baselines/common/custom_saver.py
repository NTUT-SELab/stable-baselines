import os
import tensorflow as tf
import glob

from pathlib import Path


class CustomSaver:
    def __init__(self, ckpt_path, info_path, name, id=None, mode=None):
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
        self.id = id
        self.mode = mode

    def build(self):
        self.saver = tf.train.Saver()
        sr_id = -1
        if self.mode == 1:
            latest_ckpt_id = self._get_latest_id(self.ckpt_path, self.name+'_checkpoints')
            latest_info_id = self._get_latest_id(self.info_path, self.name+'_info')
            if latest_ckpt_id != latest_info_id:
                raise ValueError("ckpt id and info id are not consistent")
            else:
                sr_id = latest_ckpt_id + 1
        elif self.mode == 0:
            sr_id = self._get_latest_id(self.ckpt_path, self.name+'_checkpoints')
            if self.id != None:
                sr_id = self.id
        self.ckpt_path = os.path.join(self.ckpt_path, "{}_{}".format(self.name+'_checkpoints', sr_id))
        self.info_path = os.path.join(self.info_path, "{}_{}".format(self.name+'_info', sr_id))
        self._create_dirs()

    def save_or_restore_ckpt(self, sess, global_step):
        if self.mode == 1:
            self.saver.save(sess, os.path.join(self.ckpt_path, 'ckpt'), global_step=global_step)
        elif self.mode == 0:
            try:
                self.saver.restore(sess, os.path.join(self.ckpt_path, 'ckpt-' + str(global_step)))
            except:
                print (f'fail to restore ckpt-{global_step}')

    def save_or_restore_info(self, save_fn, load_fn, data, name):
        if self.mode == 1:
            save_fn(os.path.join(self.info_path, name), data)
        elif self.mode == 0:
            return load_fn(os.path.join(self.info_path, name))[0]
        return None

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


        
