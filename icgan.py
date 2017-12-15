import tensorflow as tf
import numpy as np
import os
import argparse
from tensorflow.contrib.layers.python.layers import batch_norm

# hyperparameters

y_dim = 2
latent_size = 100

log_vars = []

nb_epoch = 15
nb_epoch_zy = 10

nb_sample = 40000
batch_size = 64
test_size = 10

dis_learning_rate = 0.0002
gen_learning_rate = 0.0002
eny_learning_rate = 0.001
enz_learning_rate = 0.001

TINY = 1e-8

FLAGS = None

# define placeholder
image = tf.placeholder(tf.float32, [None, 64, 64, 3])
z = tf.placeholder(tf.float32, [None, latent_size])
y = tf.placeholder(tf.float32, [None, y_dim])
train_phase = tf.placeholder(tf.bool)


# ops
def lrelu(x, alpha=0.02, name="LeakyRelu"):
    return tf.maximum(x, alpha*x)


def conv2d(x, kernel, bias, stride=2, padding="SAME"):
    x = tf.nn.conv2d(x, kernel, strides=[1, stride, stride, 1], padding=padding)
    x = tf.nn.bias_add(x, bias)
    return x


def de_conv(x, kernel, bias, output_shape, stride=2, padding="SAME"):
    x = tf.nn.conv2d_transpose(x, kernel, output_shape, [1,stride,stride,1],
                               padding=padding)
    x = tf.nn.bias_add(x, bias)
    return x


def full_connect(x, weight, bias):
    return tf.add(tf.matmul(x, weight), bias)


def batch_normal(x, scope="bn", reuse=False, istraining=True):
    return batch_norm(x, epsilon=1e-5, decay=0.9, scale=True, scope=scope, reuse=reuse, is_training=istraining,
                      updates_collections=None, center=True)


# util
def image_saver(image, fname):
    image += 1
    image *= 127.5
    enc = tf.image.encode_jpeg(image)
    fwriter = tf.write_file(fname, enc)
    return fwriter


def read_and_decode(filename, image_size, test=False):
    filename_queue = tf.train.string_input_producer([filename])
    reader = tf.TFRecordReader()
    _, example = reader.read(filename_queue)
    if test:
        features = tf.parse_single_example(example,
            features={
                "image": tf.FixedLenFeature([], tf.string)
            })
        img = tf.decode_raw(features["image"], tf.float32)
        img = tf.reshape(img, image_size)
        return img
    else:
        features = tf.parse_single_example(example,
            features={
                "label": tf.FixedLenFeature([], tf.int64),
                "image": tf.FixedLenFeature([], tf.string)
            })
        img = tf.decode_raw(features["image"], tf.float32)
        img = tf.reshape(img, image_size)
        label = tf.cast(features["label"], tf.int32)
        return img, label


def get_test_label(batch_size):
    return np.array([[1, 0]]*batch_size)


def transfer_one_hot_label(label, y_size):
    return tf.one_hot(label, y_size, on_value=1.0, off_value=0.0, axis=-1)


def get_sample(filepath, image_size, y_size):
    img, label = read_and_decode(filepath, image_size)
    label = transfer_one_hot_label(label, y_size)
    return img, label


# define variables
def gen_variables(in_size, y_size):
    weight = {
        'k1': tf.Variable(tf.truncated_normal([4, 4, 512, in_size+y_size], stddev=0.02), name="gen_w1"),
        'k2': tf.Variable(tf.truncated_normal([4, 4, 256, 512], stddev=0.02), name="gen_w2"),
        'k3': tf.Variable(tf.truncated_normal([4, 4, 128, 256], stddev=0.02), name="gen_w3"),
        'k4': tf.Variable(tf.truncated_normal([4, 4, 64, 128], stddev=0.02), name="gen_w4"),
        'k5': tf.Variable(tf.truncated_normal([4, 4, 3, 64], stddev=0.02), name="gen_w5")
    }
    bias = {
        'b1': tf.Variable(tf.zeros([512]), name='gen_b1'),
        'b2': tf.Variable(tf.zeros([256]), name='gen_b2'),
        'b3': tf.Variable(tf.zeros([128]), name='gen_b3'),
        'b4': tf.Variable(tf.zeros([64]), name='gen_b4'),
        'b5': tf.Variable(tf.zeros([3]), name='gen_b5')
    }
    return weight, bias


def dis_variables(y_size):
    weight = {
        'k1': tf.Variable(tf.truncated_normal([4, 4, 3, 64], stddev=0.02), name="dis_w1"),
        'k2': tf.Variable(tf.truncated_normal([4, 4, 64+y_size, 128], stddev=0.02), name="dis_w2"),
        'k3': tf.Variable(tf.truncated_normal([4, 4, 128, 256], stddev=0.02), name="dis_w3"),
        'k4': tf.Variable(tf.truncated_normal([4, 4, 256, 512], stddev=0.02), name="dis_w4"),
        'k5': tf.Variable(tf.truncated_normal([4, 4, 512, 1], stddev=0.02), name="dis_w5")
    }
    bias = {
        'b1': tf.Variable(tf.zeros([64]), name='dis_b1'),
        'b2': tf.Variable(tf.zeros([128]), name='dis_b2'),
        'b3': tf.Variable(tf.zeros([256]), name='dis_b3'),
        'b4': tf.Variable(tf.zeros([512]), name='dis_b4'),
        'b5': tf.Variable(tf.zeros([1]), name='dis_b5')
    }
    return weight, bias


def enz_variables(out_size):
    weight = {
        'k1': tf.Variable(tf.truncated_normal([5, 5, 3, 32], stddev=0.02), name='enz_w1'),
        'k2': tf.Variable(tf.truncated_normal([5, 5, 32, 64], stddev=0.02), name='enz_w2'),
        'k3': tf.Variable(tf.truncated_normal([5, 5, 64, 128], stddev=0.02), name='enz_w3'),
        'k4': tf.Variable(tf.truncated_normal([5, 5, 128, 256], stddev=0.02), name='enz_w4'),
        'k5': tf.Variable(tf.truncated_normal([4096, 4096], stddev=0.02), name='enz_w5'),
        'k6': tf.Variable(tf.truncated_normal([4096, out_size], stddev=0.02), name='enz_w6')
    }
    bias = {
        'b1': tf.Variable(tf.zeros([32]), name='enz_b1'),
        'b2': tf.Variable(tf.zeros([64]), name='enz_b2'),
        'b3': tf.Variable(tf.zeros([128]), name='enz_b3'),
        'b4': tf.Variable(tf.zeros([256]), name='enz_b4'),
        'b5': tf.Variable(tf.zeros([4096]), name='enz_b5'),
        'b6': tf.Variable(tf.zeros([out_size]), name='enz_b6')
    }
    return weight, bias


def eny_variables(y_size):
    weight = {
        'k1': tf.Variable(tf.truncated_normal([5, 5, 3, 32], stddev=0.02), name='eny_w1'),
        'k2': tf.Variable(tf.truncated_normal([5, 5, 32, 64], stddev=0.02), name='eny_w2'),
        'k3': tf.Variable(tf.truncated_normal([5, 5, 64, 128], stddev=0.02), name='eny_w3'),
        'k4': tf.Variable(tf.truncated_normal([5, 5, 128, 256], stddev=0.02), name='eny_w4'),
        'k5': tf.Variable(tf.truncated_normal([4096, 512], stddev=0.02), name='eny_w5'),
        'k6': tf.Variable(tf.truncated_normal([512, y_size], stddev=0.02), name='eny_w6')
    }
    bias = {
        'b1': tf.Variable(tf.zeros([32]), name='eny_b1'),
        'b2': tf.Variable(tf.zeros([64]), name='eny_b2'),
        'b3': tf.Variable(tf.zeros([128]), name='eny_b3'),
        'b4': tf.Variable(tf.zeros([256]), name='eny_b4'),
        'b5': tf.Variable(tf.zeros([512]), name='eny_b5'),
        'b6': tf.Variable(tf.zeros([y_size]), name='eny_b6')
    }
    return weight, bias

gen_w, gen_b = gen_variables(latent_size, y_dim)
dis_w, dis_b = dis_variables(y_dim)
enz_w, enz_b = enz_variables(latent_size)
eny_w, eny_b = eny_variables(y_dim)


def generator(weight, bias, z_, y_, istraining, reuse=False):
    in_size = tf.shape(z)[0]
    with tf.variable_scope('generator'):
        input_ = tf.reshape(tf.concat([z_, y_], axis=1), [in_size,1,1,-1])
        out1 = tf.nn.relu(batch_normal(de_conv(input_, weight['k1'], bias['b1'], [in_size, 4, 4, 512], padding="VALID"),
                                       scope='gen_bn1', reuse=reuse, istraining=istraining))
        out2 = tf.nn.relu(batch_normal(de_conv(out1, weight['k2'], bias['b2'], [in_size, 8, 8, 256]),
                                       scope='gen_bn2', reuse=reuse, istraining=istraining))
        out3 = tf.nn.relu(batch_normal(de_conv(out2, weight['k3'], bias['b3'], [in_size, 16, 16, 128]),
                                       scope='gen_bn3', reuse=reuse, istraining=istraining))
        out4 = tf.nn.relu(batch_normal(de_conv(out3, weight['k4'], bias['b4'], [in_size, 32, 32, 64]),
                                       scope='gen_bn4', reuse=reuse, istraining=istraining))
        out5 = tf.tanh(de_conv(out4, weight['k5'], bias['b5'], [in_size, 64, 64, 3]))
    return out5


def discriminator(weight, bias, y_, x_, istraining, reuse=False):
    with tf.variable_scope('discriminator'):
        out1 = lrelu(batch_normal(
            conv2d(x_, weight['k1'], bias['b1'], stride=2), scope='dis_bn1', istraining=istraining, reuse=reuse))
        exp_dim = tf.reshape(y_, shape=[tf.shape(y_)[0], 1, 1, tf.shape(y_)[1]]) * tf.ones(
            [tf.shape(out1)[0], tf.shape(out1)[1], tf.shape(out1)[2], tf.shape(y_)[1]])
        out1 = tf.concat([out1, exp_dim], axis=3)
        out2 = lrelu(batch_normal(
            conv2d(out1, weight['k2'], bias['b2'], stride=2), scope='dis_bn2', istraining=istraining, reuse=reuse))
        out3 = lrelu(batch_normal(
            conv2d(out2, weight['k3'], bias['b3'], stride=2), scope='dis_bn3', istraining=istraining, reuse=reuse))
        out4 = lrelu(batch_normal(
            conv2d(out3, weight['k4'], bias['b4'], stride=2), scope='dis_bn4', istraining=istraining, reuse=reuse))
        out5 = tf.sigmoid(conv2d(out4, weight['k5'], bias['b5'], stride=2, padding="VALID"))
    return out5


def en_z(weight, bias, x, istraining):
    in_size = tf.shape(x)[0]
    with tf.variable_scope('enz'):
        out1 = tf.nn.relu(batch_normal(conv2d(x, weight['k1'], bias['b1'], stride=2), scope="enz_bn1",
                                     istraining=istraining))
        out2 = tf.nn.relu(batch_normal(conv2d(out1, weight['k2'], bias['b2'], stride=2), scope="enz_bn2",
                                     istraining=istraining))
        out3 = tf.nn.relu(batch_normal(conv2d(out2, weight['k3'], bias['b3'], stride=2), scope="enz_bn3",
                                     istraining=istraining))
        out4 = tf.nn.relu(batch_normal(conv2d(out3, weight['k4'], bias['b4'], stride=2), scope="enz_bn4",
                                     istraining=istraining))
        out4 = tf.reshape(out4, [in_size, 4096])
        out5 = tf.nn.relu(batch_normal(full_connect(out4, weight['k5'], bias['b5']), scope="enz_bn5",
                                     istraining=istraining))
        out6 = full_connect(out5, weight['k6'], bias['b6'])
    return out6


def en_y(weight, bias, x, istraining):
    in_size = tf.shape(x)[0]
    with tf.variable_scope('eny'):
        out1 = tf.nn.relu(batch_normal(conv2d(x, weight['k1'], bias['b1'], stride=2), scope="eny_bn1",
                                     istraining=istraining))
        out2 = tf.nn.relu(batch_normal(conv2d(out1, weight['k2'], bias['b2'], stride=2), scope="eny_bn2",
                                     istraining=istraining))
        out3 = tf.nn.relu(batch_normal(conv2d(out2, weight['k3'], bias['b3'], stride=2), scope="eny_bn3",
                                     istraining=istraining))
        out4 = tf.nn.relu(batch_normal(conv2d(out3, weight['k4'], bias['b4'], stride=2), scope="eny_bn4",
                                     istraining=istraining))
        out4 = tf.reshape(out4, [in_size, 4096])
        out5 = tf.nn.relu(batch_normal(full_connect(out4, weight['k5'], bias['b5']), scope="eny_bn5",
                                     istraining=istraining))
        out6 = full_connect(out5, weight['k6'], bias['b6'])
    return out6


def get_z(batch_, latent_):
    return np.random.normal(0, 1.0, [batch_, latent_])


def train_gan(trainpath, gan_path, image_dir):
    print "train cgan"

    img, label = get_sample(trainpath, [64,64,3], y_dim)
    img_batch, label_batch = tf.train.shuffle_batch([img, label], batch_size=batch_size,
                                                    capacity=5 * batch_size, min_after_dequeue=0)

    fake_img = generator(gen_w, gen_b, z, y, train_phase)
    fake_img_test = generator(gen_w, gen_b, z, y, train_phase, True)

    D_fake_img = discriminator(dis_w, dis_b, y, fake_img, train_phase)
    D_real_img = discriminator(dis_w, dis_b, y, image, train_phase, True)

    G_loss = -tf.reduce_mean(tf.log(D_fake_img+TINY))
    D_loss = -tf.reduce_mean(tf.log(1.-D_fake_img+TINY) + tf.log(D_real_img+TINY))

    log_vars.append(('generator_loss', G_loss))
    log_vars.append(('discriminator_loss', D_loss))

    for k, v in log_vars:
        tf.summary.scalar(k, v)

    d_vars = [var for var in tf.trainable_variables() if 'dis' in var.name]
    g_vars = [var for var in tf.trainable_variables() if 'gen' in var.name]

    train_D = tf.train.AdamOptimizer(learning_rate=dis_learning_rate, beta1=0.5).minimize(D_loss, var_list=d_vars)
    train_G = tf.train.AdamOptimizer(learning_rate=gen_learning_rate, beta1=0.5).minimize(G_loss, var_list=g_vars)

    with tf.Session() as sess:
        sess.run(tf.global_variables_initializer())

        coord = tf.train.Coordinator()
        threads = tf.train.start_queue_runners(sess=sess, coord=coord)

        summary = tf.summary.merge_all()

        summary_writer = tf.summary.FileWriter(image_dir, sess.graph)

        step = 0

        for _ in range(nb_epoch):
            for j in range(nb_sample/batch_size):
                real_img, label = sess.run([img_batch, label_batch])
                z_in = get_z(batch_size, latent_size)

                _, summary_str = sess.run([train_D, summary], feed_dict={image: real_img, y: label, z: z_in,
                                                                         train_phase: True})
                summary_writer.add_summary(summary_str, global_step=step)
                _, summary_str = sess.run([train_G, summary], feed_dict={image: real_img, y: label, z: z_in,
                                                                         train_phase: True})
                summary_writer.add_summary(summary_str, global_step=step)

                print step

                if step%1000 == 0:
                    # img_test = read_and_decode(testpath, [64,64,3], True)
                    z_test = get_z(test_size, latent_size)
                    y_test = get_test_label(test_size)
                    gen_img = sess.run(fake_img_test, feed_dict={y: y_test, z: z_test, train_phase: False})
                    for k in range(test_size):
                        path = os.path.join(image_dir, "train_"+str(k)+"_No_"+str(step)+".jpg")
                        sess.run(image_saver(gen_img[k], path))
                step += 1
        print sess.run(gen_b["b5"])
        print sess.run(dis_b["b5"])

        save_vars = [var for var in tf.global_variables() if "gen" in var.name]
        for var in save_vars:
            print var.name
        saver = tf.train.Saver(save_vars)
        saver.save(sess, gan_path+"model.ckpt")
        coord.request_stop()
        coord.join(threads)

    print "train cgan over"


def train_enz(gan_path, z_path):

    print "train enz"

    fake_image = generator(gen_w, gen_b, z, y, train_phase)

    enz = en_z(enz_w, enz_b, image, train_phase)

    enz_loss = tf.reduce_mean(tf.square(tf.contrib.layers.flatten(enz - z)))

    summary = tf.summary.scalar("enz_loss", enz_loss)

    g_vars = [var for var in tf.global_variables() if 'gen' in var.name]

    enz_vars = [var for var in tf.trainable_variables() if "enz" in var.name]

    g_saver = tf.train.Saver(g_vars)

    enz_train = tf.train.AdamOptimizer(learning_rate=enz_learning_rate, beta1=0.5).minimize(enz_loss, var_list=enz_vars)

    with tf.Session() as sess:
        sess.run(tf.global_variables_initializer())

        ckpt = tf.train.get_checkpoint_state(gan_path)
        if ckpt and ckpt.model_checkpoint_path:
            print "gan restore"
            g_saver.restore(sess, ckpt.model_checkpoint_path)

        step = 0

        summary_writer = tf.summary.FileWriter(z_path, sess.graph)

        coord = tf.train.Coordinator()
        threads = tf.train.start_queue_runners(sess=sess, coord=coord)

        for _ in range(nb_epoch_zy):
            for _ in range(nb_sample/batch_size):
                label = get_test_label(batch_size)
                z_in = get_z(batch_size, latent_size)

                new_image = sess.run(fake_image, feed_dict={z: z_in, y: label, train_phase: False})

                _, summary_str = sess.run([enz_train, summary], feed_dict={y: label, z: z_in, image:new_image,
                                                                           train_phase: True})
                summary_writer.add_summary(summary_str, global_step=step)

                if step%1000 == 0:
                    loss = sess.run(enz_loss, feed_dict={y: label, z: z_in, image: new_image, train_phase: False})
                    print "step %d have loss %.7f" % (step, loss)
                step += 1
        save_vars = [var for var in tf.global_variables() if "enz" in var.name]
        z_saver = tf.train.Saver(save_vars)
        z_saver.save(sess, z_path+'model.ckpt')
        coord.request_stop()
        coord.join(threads)
    print "train enz over"


def train_eny(train_eny_path, y_path):

    print "train eny"

    img, label = get_sample(train_eny_path, [64, 64, 3], y_dim)

    img_batch, label_batch = tf.train.shuffle_batch([img, label], batch_size=batch_size,
                                                    capacity=5 * batch_size, min_after_dequeue=0)

    eny = en_y(eny_w, eny_b, image, train_phase)

    loss_y = tf.reduce_mean(tf.square(eny - y))

    eny_vars = [var for var in tf.trainable_variables()  if "eny" in var.name]

    # log_vars_z.append(('eny_loss_loss', loss_y))
    #
    # for k, v in log_vars:
    #     tf.summary.scalar(k, v)

    summary = tf.summary.scalar("eny_loss", loss_y)

    eny_train = tf.train.AdamOptimizer(learning_rate=eny_learning_rate, beta1=0.5).minimize(loss_y, var_list=eny_vars)

    with tf.Session() as sess:
        sess.run(tf.global_variables_initializer())

        step = 0

        coord = tf.train.Coordinator()
        threads = tf.train.start_queue_runners(sess=sess, coord=coord)

        summary_writer = tf.summary.FileWriter(y_path, sess.graph)

        z_in = get_z(batch_size, latent_size)

        for _ in range(nb_epoch_zy):
            for _ in range(nb_sample/batch_size):
                image_in, label_in = sess.run([img_batch, label_batch])

                _, summary_str = sess.run([eny_train, summary], feed_dict={image:image_in, y:label_in, z: z_in, train_phase: True})
                summary_writer.add_summary(summary_str, global_step=step)

                if step % 1000 == 0:
                    loss = sess.run(loss_y, feed_dict={image:image_in, y:label_in, train_phase: False})
                    print "step %d have loss %.7f"%(step, loss)
                step += 1

        save_vars = [var for var in tf.global_variables() if "eny" in var.name]
        saver_y = tf.train.Saver(save_vars)
        saver_y.save(sess, y_path+"model.ckpt")

        coord.request_stop()
        coord.join(threads)
    print "train eny over"


def test(testpath, y_path, z_path, g_path, image_dir):

    img_test = read_and_decode(testpath, [64, 64, 3], True)

    img_test_in = tf.train.shuffle_batch([img_test], batch_size=test_size,
                                                    capacity=test_size, min_after_dequeue=0)

    z_ = en_z(enz_w, enz_b, image, train_phase)
    y_ = en_y(eny_w, eny_b, image, train_phase)

    gen_img = generator(gen_w, gen_b, z, y, train_phase)

    g_vars = [var for var in tf.global_variables() if 'gen' in var.name]
    eny_vars = [var for var in tf.global_variables() if "eny" in var.name]
    enz_vars = [var for var in tf.global_variables() if "enz" in var.name]

    y_saver = tf.train.Saver(eny_vars)
    g_saver = tf.train.Saver(g_vars)
    z_saver = tf.train.Saver(enz_vars)

    new_label = get_test_label(test_size)

    with tf.Session() as sess:
        # sess.run(tf.global_variables_initializer())

        cpkt_y = tf.train.get_checkpoint_state(y_path)
        if cpkt_y and cpkt_y.model_checkpoint_path:
            print "restore y"
            y_saver.restore(sess, cpkt_y.model_checkpoint_path)

        cpkt_z = tf.train.get_checkpoint_state(z_path)
        if cpkt_z and cpkt_z.model_checkpoint_path:
            print "restore z"
            z_saver.restore(sess, cpkt_z.model_checkpoint_path)

        cpkt_g = tf.train.get_checkpoint_state(g_path)
        if cpkt_g and cpkt_g.model_checkpoint_path:
            print "restore gan"
            g_saver.restore(sess, cpkt_g.model_checkpoint_path)

        print "read over"

        coord = tf.train.Coordinator()
        threads = tf.train.start_queue_runners(sess=sess, coord=coord)

        img_in = sess.run(img_test_in)

        y_in = sess.run(y_, feed_dict={image: img_in, train_phase: False})
        z_in = sess.run(z_, feed_dict={image: img_in, train_phase: False})

        new_image = sess.run(gen_img, feed_dict={image: img_in, y: new_label, z: z_in, train_phase: False})

        for i in range(test_size):
            path = os.path.join(image_dir, "fake_man_"+"test_"+str(i)+".jpg")
            sess.run(image_saver(new_image[i], path))
        for i in range(test_size):
            path = os.path.join(image_dir, "groundtruth_"+"test_"+str(i)+".jpg")
            sess.run(image_saver(img_in[i], path))
        coord.request_stop()
        coord.join(threads)
        print "test over"

## ATTENTION
train_path = "validation_64.tfrecords"
test_path = "test_64.tfrecords"

model_path = os.path.join(os.getcwd()+"/model_gan/")
y_model_path = os.path.join(os.getcwd()+"/model_y/")
z_model_path = os.path.join(os.getcwd()+"/model_z/")


log_dir = os.path.join(os.getcwd()+"/logs/")
train_enz(model_path, z_model_path)
train_eny(train_path, y_model_path)
train_gan(train_path, model_path, log_dir)

test(test_path, y_model_path, z_model_path, model_path, log_dir)


# def main(_):
#     train_path = os.path.join(FLAGS.buckets, "train_64.tfrecords")
#     test_path = os.path.join(FLAGS.buckets, "test_64.tfrecords")
#     model_path = os.path.join(FLAGS.checkpointDir, "model_gan/")
#     y_model_path = os.path.join(FLAGS.checkpointDir, "model_y/")
#     z_model_path = os.path.join(FLAGS.checkpointDir, "model_z/")
#     log_dir = os.path.join(FLAGS.checkpointDir, "logs")
#     # train_eny(train_path, y_model_path)
#     # train_enz(model_path, z_model_path)
#     # train_gan(train_path, model_path, log_dir)
#     test(test_path, y_model_path, z_model_path, model_path, log_dir)

# if __name__ == '__main__':
#     parser = argparse.ArgumentParser()
#     # input path
#     parser.add_argument('--buckets', type=str, default='',
#                         help='input data path')
#     # output path
#     parser.add_argument('--checkpointDir', type=str, default='',
#                         help='output model path')
#     FLAGS, _ = parser.parse_known_args()
#     tf.app.run(main=main)