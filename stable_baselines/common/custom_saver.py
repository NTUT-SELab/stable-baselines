import os
import tensorflow as tf


class CustomSaver:
    def __init__(self, ckpt_path, info_path, save_n_batch=0, restore_batch=0):
        self.ckpt_path = ckpt_path
        self.info_path = info_path
        self.save_n_batch = save_n_batch
        self.restore_batch = restore_batch
        
    def save(self, graph, sess, batch_num, total_batches):
        save_freq = total_batches // self.save_n_batch
        if batch_num % save_freq == 0:
            with graph.as_default():
                saver = tf.train.Saver()
                saver.save(sess, os.path.join(self.ckpt_path, 'ckpt'), global_step=batch_num)
                print ("Saved!" + str(batch_num))

    def restore(self, graph, sess, batch_num):
        if batch_num >= self.restore_batch:
            with graph.as_default():
                saver = tf.train.Saver()
                try:
                    saver.restore(sess, os.path.join(self.ckpt_path, 'ckpt-' + str(batch_num)))
                    print ("Restored!" + str(batch_num))
                except:
                    pass

    def save_info(self, save_fn, data, batch_num, name):
        save_fn(os.path.join(self.info_path, name+'_batch'+str(batch_num)), data)
        print ("Info Saved " + str(batch_num))

    def restore_info(self, load_fn, batch_num, name):
        if batch_num >= self.restore_batch:
            print ("Info Restored " + str(batch_num))
            return load_fn(os.path.join(self.info_path, name+'_batch'+str(batch_num)))
        return(None, None)

            


        
