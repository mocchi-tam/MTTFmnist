from math import ceil
import tensorflow as tf

class MTTFNet():
    def __init__(self, Network_layers):
        self.Network_layers = Network_layers
        self.initial_W = 0.1
        
        ## slope(leaky_relu)
        self.LRelu_alpha = 0.2
    
    def set_keep_prob(self, keep_prob):
        self.keep_prob = keep_prob
    
    # 損失関数定義
    def loss_func(self, y, t):
        loss = tf.nn.softmax_cross_entropy_with_logits(logits=y, labels=t)
        return tf.reduce_mean(loss)
        
    def make(self, x):
        if len(x.shape) == 4:
            img_x = int(x.shape[1])
            img_y = int(x.shape[2])
            img_ch = int(x.shape[3])
        else:
            img_x = 1
            img_y = 1
            img_ch = int(x.shape[1])
        
        h = x
        n_layer = 0
        ## ニューラルネット構成
        for hierarchy in self.Network_layers:
            layer_name = 'layer' + str(n_layer)
            n_layer += 1
            with tf.variable_scope(layer_name):
                layer = hierarchy['layer']
                ch = hierarchy['ch']
                activation = hierarchy['activation']
                total_binding_ch = img_x*img_y*img_ch
                
                b = tf.Variable(tf.constant(self.initial_W, shape=[ch]), name='b')
                if layer == 'CNN':
                    ksize = hierarchy['ksize']
                    stride = hierarchy['stride']
                    
                    #### 重みの初期化
                    kernel_ch = ksize[0]*ksize[1]*img_ch
                    W = tf.Variable(tf.truncated_normal([kernel_ch*ch], stddev=self.initial_W), name='W')
                    W_conv = tf.reshape(W, [ksize[0],ksize[1],img_ch,ch])
                    
                    #### 畳み込み層
                    h = tf.nn.conv2d(h, W_conv, strides=[1, stride[0], stride[1], 1], padding='SAME') + b
                    img_x = ceil(float(img_x) / float(stride[0]))
                    img_y = ceil(float(img_y) / float(stride[1]))
                    img_ch = ch
                    
                elif layer == 'Linear':
                    use_dropout = hierarchy['dropout']
                    
                    #### 重みの初期化
                    W = tf.Variable(tf.truncated_normal([total_binding_ch*ch], stddev=self.initial_W), name='W')
                    W_lin = tf.reshape(W, [total_binding_ch,ch])
                    
                    ### 全結合層
                    h = tf.reshape(h, [-1, total_binding_ch])
                    h = tf.matmul(h, W_lin) + b
                    img_x = 1
                    img_y = 1
                    img_ch = ch
                    
                    ### ドロップアウト
                    if use_dropout:
                        h = tf.nn.dropout(h, self.keep_prob)
                    
                ### 活性化関数
                if activation == 'Relu':
                    h = tf.nn.relu(h)
                elif activation == 'LRelu':
                    # leaky_relu
                    h = tf.maximum(h, self.LRelu_alpha*h)
                
        print('MTINFO: successfully constructed the neural network')
        return h