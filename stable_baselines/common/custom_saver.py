import os
import tensorflow as tf


class CustomSaver:
    def __init__(self, ckpt_path, info_path, save_freq=0):
        self.ckpt_path = ckpt_path
        self.info_path = info_path
        self.save_freq = save_freq
        self.step = 0

    def save(self, graph, sess, global_step):
        if global_step == self.step:
            with graph.as_default():
                with tf.variable_scope('', reuse=True):
                    saver = tf.train.Saver()
                    saver.save(sess, os.path.join(self.ckpt_path, 'ckpt'), global_step=global_step)
                    self.step += self.save_freq

    def restore(self, graph, sess, global_step):
        if global_step == self.step:
            with graph.as_default():
                with tf.variable_scope('', reuse=True):
                    saver = tf.train.Saver()
                    try:
                        saver.restore(sess, os.path.join(self.ckpt_path, 'ckpt-' + str(global_step)))
                        self.step += self.save_freq
                    except:
                        print (f'fail to restore ckpt-{global_step}')

    def save_info(self, save_fn, data, global_step, name):
        save_fn(os.path.join(self.info_path, name+'_step_'+str(global_step)), data)

    def restore_info(self, load_fn, global_step, name):
        return load_fn(os.path.join(self.info_path, name+'_step_'+str(global_step)))

            


        
