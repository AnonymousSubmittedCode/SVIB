import tensorflow as tf
from tensorflow.contrib.solvers.python.ops.util import l2norm
import numpy as np
import cv2 as cv
from baselines.a2c.utils import conv, lstm, lnlstm, conv_to_fc, fc, ortho_init

def grad_clip(loss, max_grad_norm, scope_list):
    '''
    :param loss:
    :param params:
    :param max_grad_norm:
    :param scope: a list consist of variable scopes
    :return:
    '''
    params_list = []
    for scope in scope_list:
        List = tf.get_collection(tf.GraphKeys.TRAINABLE_VARIABLES, scope = scope)
        print(len(List))
        params_list += List
    grads = tf.gradients(loss, params_list)
    # for i, grad in enumerate(grads):
    #     if grad is None:
    #         grads[i] = tf.zeros(shape=params_list[i].get_shape(), dtype=params_list[i].dtype)
    global_norm = 0.
    if max_grad_norm is not None:
        grads, grad_norm = tf.clip_by_global_norm(grads, max_grad_norm)
        global_norm = tf.sqrt(sum([l2norm(t) ** 2 for t in grads]))
    grads = list(zip(grads, params_list))
    return grads, global_norm

def grad_clip_joint(loss_joint, max_grad_norm, scope_list_joint):
    '''
    :param loss_joint: [loss1, loss2, loss3, ...]
    :param max_grad_norm:
    :param scope_list_joint: [scope_list1(with respect to loss1), scope_list2, scope_list3, ..,]
    :return:
    '''
    grads_joint = []
    params_list_joint = []
    seg_points = [int(0)]
    for i, loss in  enumerate(loss_joint):
        params_list = []
        for scope in scope_list_joint[i]:
            List = tf.get_collection(tf.GraphKeys.TRAINABLE_VARIABLES, scope=scope)
            print(len(List))
            params_list += List
        grads_joint += tf.gradients(loss, params_list)
        params_list_joint += params_list
        seg_points.append(len(params_list))
    if max_grad_norm is not None:
        grads_joint, grad_norm = tf.clip_by_global_norm(grads_joint, max_grad_norm)
    grads_joint_return = []
    global_norm_return = []
    for i in range(len(seg_points)-1):
        grads = grads_joint[seg_points[i]:seg_points[i+1]]
        params_list = params_list_joint[seg_points[i]:seg_points[i+1]]
        global_norm = tf.sqrt(sum([l2norm(t) ** 2 for t in grads]))
        grads = list(zip(grads, params_list))
        grads_joint_return.append(grads)
        global_norm_return.append(global_norm)
    return grads_joint_return, global_norm_return

def l2norm1(a):
    '''
    :param a: shape = [bs, dim]
    :return: l2norm of a along axis = 1, shape = [bs]
    '''
    return tf.sqrt(tf.reduce_sum(tf.square(a), 1))

def compute_cosine(a, b):
    '''
    :param a: shape = [bs, dim]
    :param b: shape = [bs, dim]
    :return: cosine similarity along axis=1
    '''
    numer = tf.reduce_sum(a*b, axis=1)
    deno = tf.stop_gradient(l2norm1(a)*l2norm1(b))
    return numer / (deno + 1e-3)

def np_l2norm1(a):
    return np.sqrt(np.sum(np.square(a), axis=1))

def np_compute_cosine(a, b):
    numer = np.sum(a*b, axis=1)
    deno = np_l2norm1(a) * np_l2norm1(b)
    return numer / (deno + 1e-3)

def explained_variance(ypred,y):
    """
    Computes fraction of variance that ypred explains about y.
    Returns 1 - Var[y-ypred] / Var[y]

    interpretation:
        ev=0  =>  might as well have predicted zero
        ev=1  =>  perfect prediction
        ev<0  =>  worse than just predicting zero

    """
    assert y.ndim == 1 and ypred.ndim == 1
    vary = np.var(y)
    return np.nan if vary==0 else 1 - np.var(y-ypred)/vary

def batch_norm(x, scope, nstep, var_eps = 1e-3):
    '''
    :param x:
    :param scope:
    :param worker_step: time step of worker
    :param var_eps:
    :return:
    '''
    if nstep == 1:
        return x
    shape = list(x.get_shape())
    mean, var = tf.nn.moments(x, axes = list(range(len(shape) - 1)))
    depth = shape[-1].value
    with tf.variable_scope(scope):
        scale = tf.get_variable(shape = [depth], name = 'scale', dtype = tf.float32)
        offset = tf.get_variable(shape = [depth], name = 'offset', dtype = tf.float32)
    x_ = scale * ((x - mean) / tf.sqrt(var + var_eps)) + offset
    return x_

def show_img(img, scope):
    # print(img.shape)
    # print((img[:,:,0:1] == img[:,:,1:2]).all())
    # print(img.shape)
    cv.namedWindow(scope, cv.WINDOW_AUTOSIZE)
    cv.imshow(scope, cv.resize(img[:,:,0], (3*84, 3*84)))
    cv.waitKey(1)

def conv_unequ_size(x, scope, *, nf, rf, stride, pad='VALID', init_scale=1.0, data_format='NHWC'):
    '''
    :param x:
    :param scope:
    :param nf:
    :param rf: a 2-element list, [kernel_h, kernel_w]
    :param stride: a 2-element list, [h, w]
    :param pad:
    :param init_scale:
    :param data_format:
    :return:
    '''
    if data_format == 'NHWC':
        channel_ax = 3
        strides = [1, stride[0], stride[1], 1]
        bshape = [1, 1, 1, nf]
    elif data_format == 'NCHW':
        channel_ax = 1
        strides = [1, 1, stride[0], stride[1]]
        bshape = [1, nf, 1, 1]
    else:
        raise NotImplementedError
    nin = x.get_shape()[channel_ax].value
    wshape = [rf[0], rf[1], nin, nf]
    with tf.variable_scope(scope):
        w = tf.get_variable("w", wshape, initializer=ortho_init(init_scale))
        b = tf.get_variable("b", [1, nf, 1, 1], initializer=tf.constant_initializer(0.0))
        if data_format == 'NHWC': b = tf.reshape(b, bshape)
        return b + tf.nn.conv2d(x, w, strides=strides, padding=pad, data_format=data_format)

def deconv(x, scope, *, nf, rf, stride, output_shape, pad='VALID', init_scale=1.0, data_format='NHWC'):
    if data_format == 'NHWC':
        channel_ax = 3
        strides = [1, stride, stride, 1]
        bshape = [1, 1, 1, nf]
    elif data_format == 'NCHW':
        channel_ax = 1
        strides = [1, 1, stride, stride]
        bshape = [1, nf, 1, 1]
    else:
        raise NotImplementedError
    nin = x.get_shape()[channel_ax].value
    wshape = [rf, rf, nf, nin]
    with tf.variable_scope(scope):
        w = tf.get_variable("w", wshape, initializer=ortho_init(init_scale))
        b = tf.get_variable("b", [1, nf, 1, 1], initializer=tf.constant_initializer(0.0))
        if data_format == 'NHWC': b = tf.reshape(b, bshape)
        return b + tf.nn.conv2d_transpose(x, w, output_shape, strides=strides, padding=pad, data_format=data_format)

def deconv_unequ_size(x, scope, *, nf, rf, stride, output_shape, pad='VALID', init_scale=1.0, data_format='NHWC'):
    '''
    :param x:
    :param scope:
    :param nf:
    :param rf: a 2-element list, [kernel_h, kernel_w]
    :param stride: a 2-element list, [h, w]
    :param pad:
    :param init_scale:
    :param data_format:
    :return:
    '''
    if data_format == 'NHWC':
        channel_ax = 3
        strides = [1, stride[0], stride[1], 1]
        bshape = [1, 1, 1, nf]
    elif data_format == 'NCHW':
        channel_ax = 1
        strides = [1, 1, stride[0], stride[1]]
        bshape = [1, nf, 1, 1]
    else:
        raise NotImplementedError
    nin = x.get_shape()[channel_ax].value
    wshape = [rf[0], rf[1], nf, nin]
    with tf.variable_scope(scope):
        w = tf.get_variable("w", wshape, initializer=ortho_init(init_scale))
        b = tf.get_variable("b", [1, nf, 1, 1], initializer=tf.constant_initializer(0.0))
        if data_format == 'NHWC': b = tf.reshape(b, bshape)
        return b + tf.nn.conv2d_transpose(x, w, output_shape, strides=strides, padding=pad, data_format=data_format)

def compute_ir(h, goal):
    ir = 1. * (-np_l2norm1(h - goal) / (np_l2norm1(goal) + np_l2norm1(h)))
    mask = (ir < -0.03).astype(ir.dtype)
    ir = mask*ir + (1-mask)*(ir+10.)
    return ir

def obs_preprocess(obs):
    '''
    :param obs: input shape = [bs, h, w, c=3*x]
    :return: output shape = [bs, 84, 84, c = x]
    '''
    frame_num = obs.shape[-1] // 3
    obs_list = [obs[:, :, :, i:i+3] for i in range(frame_num)]
    obs_gray_list = [single_obs.mean(-1, keepdims=True).astype(np.float32) for single_obs in obs_list]
    obs_gray = np.concatenate(tuple(obs_gray_list), axis=-1) / 255.
    obs_ = []
    for i in range(obs.shape[0]):
        obs_.append(cv.resize(obs_gray[i], (84, 84)))
    return np.asarray(obs_, dtype=np.float32)

def noise_and_argmax(logits):
    # Add noise then take the argmax
    p = tf.nn.softmax(logits=logits)
    a = tf.multinomial(p, num_samples=1)
    # noise = tf.random_uniform(tf.shape(logits))
    # return tf.argmax(logits, 1)
    return tf.squeeze(a, axis=-1)

def dense_layers(h, cell, name, activ=tf.nn.relu, num_layers=1):
    h_ = h
    for i in range(num_layers):
        h_ = activ(fc(h_, name+str(i+1), nh=cell))
    return h_

def generate_noise(size):
    '''
    :param size: a integer tuple
    :return:
    '''
    return np.random.normal(0., 0.1, size=size)

def tf_l2norm(X, axis=-1, keep_dims=False):
    return tf.sqrt(tf.reduce_sum(X*X, axis=axis, keep_dims=keep_dims))

def tf_normalize(X, axis=0):
    mean, var = tf.nn.moments(X, axes=axis, keep_dims=True)
    return -(X - tf.stop_gradient(mean)) / (tf.sqrt(tf.stop_gradient(var)) + 1e-3)

def mse(pred, target):
    return tf.square(pred-target)/2.