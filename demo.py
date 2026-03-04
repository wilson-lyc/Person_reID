import argparse
import scipy.io
import torch
import numpy as np
import os
from torchvision import datasets
import matplotlib
matplotlib.use('agg')
import matplotlib.pyplot as plt
#######################################################################
# Evaluate
parser = argparse.ArgumentParser(description='Demo')
parser.add_argument('--query_index', default=777, type=int, help='test_image_index')
parser.add_argument('--test_dir',default='../Market/pytorch',type=str, help='./test_data')
parser.add_argument('--result_mat', default='pytorch_result.mat', type=str, help='path to pytorch result mat')
opts = parser.parse_args()

data_dir = opts.test_dir
image_datasets = {x: datasets.ImageFolder( os.path.join(data_dir,x) ) for x in ['gallery','query']}

#####################################################################
#Show result
def imshow(path, title=None):
    """Imshow for Tensor."""
    im = plt.imread(path)
    plt.imshow(im)
    if title is not None:
        plt.title(title)
    plt.pause(0.001)  # pause a bit so that plots are updated

######################################################################
result = scipy.io.loadmat(opts.result_mat)
query_feature = torch.FloatTensor(result['query_f'])
query_cam = result['query_cam'][0]
query_label = result['query_label'][0]
gallery_feature = torch.FloatTensor(result['gallery_f'])
gallery_cam = result['gallery_cam'][0]
gallery_label = result['gallery_label'][0]

result_dir = os.path.dirname(opts.result_mat) or '.'
result_base = os.path.basename(opts.result_mat)
model_id = os.path.basename(os.path.normpath(result_dir)) if result_dir not in ('', '.') else os.path.splitext(result_base)[0]
dataset_suffix = ''
if result_base.startswith('pytorch_result_') and result_base.endswith('.mat'):
    dataset_suffix = result_base[len('pytorch_result_'):-len('.mat')]

if dataset_suffix:
    multi_mat_path = os.path.join(result_dir, f'multi_query_{dataset_suffix}.mat')
else:
    multi_mat_path = os.path.join(result_dir, 'multi_query.mat')

if not os.path.isfile(multi_mat_path):
    fallback_multi = os.path.join(result_dir, 'multi_query.mat')
    if os.path.isfile(fallback_multi):
        multi_mat_path = fallback_multi

multi = os.path.isfile(multi_mat_path)

if multi:
    m_result = scipy.io.loadmat(multi_mat_path)
    mquery_feature = torch.FloatTensor(m_result['mquery_f'])
    mquery_cam = m_result['mquery_cam'][0]
    mquery_label = m_result['mquery_label'][0]
    mquery_feature = mquery_feature.cuda()

query_feature = query_feature.cuda()
gallery_feature = gallery_feature.cuda()

#######################################################################
# sort the images
def sort_img(qf, ql, qc, gf, gl, gc):
    query = qf.view(-1,1)
    # print(query.shape)
    score = torch.mm(gf,query)
    score = score.squeeze(1).cpu()
    score = score.numpy()
    # predict index
    index = np.argsort(score)  #from small to large
    index = index[::-1]
    # index = index[0:2000]
    # good index
    query_index = np.argwhere(gl==ql)
    #same camera
    camera_index = np.argwhere(gc==qc)

    #good_index = np.setdiff1d(query_index, camera_index, assume_unique=True)
    junk_index1 = np.argwhere(gl==-1)
    junk_index2 = np.intersect1d(query_index, camera_index)
    junk_index = np.append(junk_index2, junk_index1) 

    mask = np.in1d(index, junk_index, invert=True)
    index = index[mask]
    return index

i = opts.query_index
if i < 0 or i >= len(query_label):
    raise ValueError(f"query_index out of range: {i}, expected 0~{len(query_label)-1}")
index = sort_img(query_feature[i],query_label[i],query_cam[i],gallery_feature,gallery_label,gallery_cam)

########################################################################
# Visualize the rank result

query_path, _ = image_datasets['query'].imgs[i]
query_pid = int(query_label[i])
query_camera = int(query_cam[i])
print(f"result_mat: {opts.result_mat}")
print(f"query_index: {i}")
print(f"query_id: {query_pid}")
print(f"query_cam: {query_camera}")
print(f"query_img: {query_path}")
print('Top 10 images are as follow:')
fig = plt.figure(figsize=(14, 3.2))
try: # Visualize Ranking Result 
    # Graphical User Interface is needed
    ax = plt.subplot(1,11,1)
    ax.axis('off')
    imshow(query_path,'query')
    ax.text(0.5, -0.08, f'ID: {query_pid}', transform=ax.transAxes, ha='center', va='top')
    for i in range(10):
        ax = plt.subplot(1,11,i+2)
        ax.axis('off')
        img_path, _ = image_datasets['gallery'].imgs[index[i]]
        label = int(gallery_label[index[i]])
        imshow(img_path)
        id_color = 'green' if label == query_pid else 'red'
        ax.text(0.5, -0.08, f'ID: {label}', transform=ax.transAxes, ha='center', va='top', color=id_color)
        if label == query_pid:
            ax.set_title('%d'%(i+1))
        else:
            ax.set_title('%d'%(i+1))
        print(img_path)
except RuntimeError:
    for i in range(10):
        img_path = image_datasets.imgs[index[i]]
        print(img_path[0])
    print('If you want to see the visualization of the ranking result, graphical user interface is needed.')

fig.subplots_adjust(left=0.01, right=0.99, top=0.92, bottom=0.16, wspace=0.05)
output_filename = f"{model_id}_{i}_{query_pid}.png"
output_path = os.path.join(result_dir, output_filename)
fig.savefig(output_path, bbox_inches='tight', pad_inches=0.03)
print(f"saved_figure: {output_path}")
