import argparse

import tensorflow as tf
import tfnet

# ニューラルネットを定義
Network_layers = [{'layer':'CNN', 'ch':8, 'ksize':(3,3), 'stride':(2,2), 'activation':'LRelu'},
                   {'layer':'CNN', 'ch':16, 'ksize':(3,3), 'stride':(2,2), 'activation':'LRelu'},
                   {'layer':'CNN', 'ch':32, 'ksize':(3,3), 'stride':(2,2), 'activation':'LRelu'},
                   {'layer':'Linear', 'ch':32, 'dropout':True, 'activation':'LRelu'},
                   {'layer':'Linear', 'ch':10, 'dropout':True, 'activation':None},
                   ]

def main():
    parser = argparse.ArgumentParser(description='Tensorflow example: MNIST')
    parser.add_argument('--batchsize', '-b', type=int, default=256,
                        help='Number of images in each mini-batch')
    parser.add_argument('--epoch', '-e', type=int, default=3,
                        help='Number of sweeps over the dataset to train')
    parser.add_argument('--gpu', '-g', type=int, default=-1,
                        help='GPU ID (negative value indicates CPU)')
    parser.add_argument('--train', '-t', type=int, default=1,
                        help='If negative, skip training')
    parser.add_argument('--resume', '-r', type=int, default=-1,
                        help='If positive, resume the training from snapshot')
    args = parser.parse_args()
    
    print('GPU: {}'.format(args.gpu))
    print('# Minibatch-size: {}'.format(args.batchsize))
    print('# epoch: {}'.format(args.epoch))
    
    flag_train = False if args.train < 0 else True
    flag_resum = False if args.resume < 0 else True
    n_epoch = args.epoch if flag_train == True else 1
    
    tsm = MTTFModel(args.gpu, flag_train, flag_resum, n_epoch, args.batchsize)
    tsm.run()

class MTTFModel():
    def __init__(self, gpu, flag_train, flag_resum, n_epoch, batchsize):
        # 初期化
        tf.reset_default_graph()
        
        # CPU/GPU の切り替え 未実装
        #device = '/cpu:0' if gpu < 0 else '/gpu:' + str(gpu)
        
        self.device = device
        self.n_epoch = n_epoch
        self.batchsize = batchsize
        self.flag_train = flag_train
        self.flag_resum = flag_resum
        
        learning_rate = 1e-3
        self.x = tf.placeholder(tf.float32, shape=[None, 784])
        self.t = tf.placeholder(tf.float32, shape=[None, 10])
        self.keep_prob = tf.placeholder(tf.float32)
        
        x_image = tf.reshape(self.x, [-1,28,28,1])
        
        # ニューラルネットを構築
        net = tfnet.MTTFNet(Network_layers)
        net.set_keep_prob(self.keep_prob)
        self.y = net.make(x_image)
        loss = net.loss_func(self.y,self.t)
        
        # Optimizer(=Adam)
        self.train_step = tf.train.AdamOptimizer(learning_rate).minimize(loss)
            
        # Accuracy setup
        correct_prediction = tf.equal(tf.argmax(self.y,1), tf.argmax(self.t,1))
        self.accuracy = tf.reduce_mean(tf.cast(correct_prediction, tf.float32))
    
    def get_acc(self, x,t):
        val_accuracy = self.accuracy.eval(feed_dict={self.x:x,self.t:t,self.keep_prob:1.0})
        return val_accuracy
    
    def run(self):
        # MNIST を取得
        #from tensorflow.contrib.learn.python.learn.datasets.mnist import read_data_sets
        data = tf.examples.tutorials.mnist.input_data.read_data_sets('MNIST_data/', one_hot=True)
        
        saver = tf.train.Saver()
        with tf.Session() as sess:
            init = tf.global_variables_initializer()
            sess.run(init)
            
            # 保存モデルの読み込み
            ckpt = tf.train.get_checkpoint_state('model/')
            if ckpt and self.flag_resum:
                try:
                    last_model = ckpt.model_checkpoint_path
                    saver.restore(sess, last_model)
                except:
                    print('MTINFO: The restored model is invalid')
            else:
                print('MTINFO: All paramteres are initalized')
            
            # 学習フェイズ
            if self.flag_train:
                epoch = 1
                epochs_completed = 0
                batch_count = 0
                tot_acc = 0
                
                while(epochs_completed < self.n_epoch):
                    batch = data.train.next_batch(self.batchsize, shuffle=True)
                    if epoch == epochs_completed:
                        print('MTINFO: train epoch %d > accuracy %g'%(epoch, tot_acc/batch_count))
                        epoch += 1
                        batch_count = 0
                        tot_acc = 0

                    self.train_step.run(feed_dict={self.x: batch[0],self.t: batch[1],self.keep_prob: 0.5})
                    epochs_completed = data.train._epochs_completed
                    tot_acc += self.get_acc(batch[0],batch[1])
                    batch_count += 1
                else:
                    print('MTINFO: train epoch %d > accuracy %g'%(epoch, tot_acc/batch_count))
            
            # テストフェイズ
            tot_acc = self.get_acc(data.test.images, data.test.labels)
            print('MTINFO: test accuracy %g'%(tot_acc))
            
            saver.save(sess, 'model/model.ckpt')
            
if __name__ == '__main__':
    main()
