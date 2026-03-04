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
model_name = os.path.basename(os.path.normpath(result_dir)) if result_dir not in ('', '.') else os.path.splitext(result_base)[0]
name_parts = model_name.split('_')
if len(name_parts) >= 2 and len(name_parts[0]) == 4 and len(name_parts[1]) == 6 and name_parts[0].isdigit() and name_parts[1].isdigit():
    model_id = f"{name_parts[0]}_{name_parts[1]}"
else:
    model_id = model_name
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

query_index = opts.query_index
if query_index < 0 or query_index >= len(query_label):
    raise ValueError(f"query_index out of range: {query_index}, expected 0~{len(query_label)-1}")
index = sort_img(
    query_feature[query_index],
    query_label[query_index],
    query_cam[query_index],
    gallery_feature,
    gallery_label,
    gallery_cam,
)

########################################################################
# Visualize the rank result

query_path, _ = image_datasets['query'].imgs[query_index]
query_pid = int(query_label[query_index])
query_camera = int(query_cam[query_index])
same_id_idx = np.argwhere(gallery_label == query_pid).flatten()
same_cam_idx = np.argwhere(gallery_cam == query_camera).flatten()
target_idx = np.setdiff1d(same_id_idx, same_cam_idx, assume_unique=False)
topk_count = min(10, len(index))
ncols = 11
target_rows = max(1, int(np.ceil(len(target_idx) / ncols)))
total_rows = 1 + target_rows
output_filename = f"{model_id}_{query_index}_{query_pid}.png"
print(f"result_mat: {opts.result_mat}")
print(f"query_index: {query_index}")
print(f"query_id: {query_pid}")
print(f"query_cam: {query_camera}")
print(f"query_img: {query_path}")
print(f"target_count: {len(target_idx)}")
print("target_imgs:")
for target_gallery_idx in target_idx:
    target_img_path, _ = image_datasets['gallery'].imgs[int(target_gallery_idx)]
    print(target_img_path)
print('Top 10 images are as follow:')
fig = plt.figure(figsize=(max(14, ncols * 1.25), 3.6 + target_rows * 2.9))
fig.subplots_adjust(left=0.01, right=0.99, top=0.92, bottom=0.08, wspace=0.05, hspace=0.35)
try: # Visualize Ranking Result 
    # Graphical User Interface is needed
    query_ax = plt.subplot(total_rows, ncols, 1)
    query_ax.axis('off')
    imshow(query_path,'query')
    query_ax.text(0.5, -0.08, f'ID: {query_pid}', transform=query_ax.transAxes, ha='center', va='top')
    for rank_i in range(topk_count):
        ax = plt.subplot(total_rows, ncols, rank_i + 2)
        ax.axis('off')
        img_path, _ = image_datasets['gallery'].imgs[index[rank_i]]
        label = int(gallery_label[index[rank_i]])
        imshow(img_path)
        id_color = 'green' if label == query_pid else 'red'
        ax.text(0.5, -0.08, f'ID: {label}', transform=ax.transAxes, ha='center', va='top', color=id_color)
        ax.set_title('%d'%(rank_i+1))
        print(img_path)
    for empty_col in range(topk_count + 1, ncols):
        ax = plt.subplot(total_rows, ncols, empty_col + 1)
        ax.axis('off')

    first_target_ax = None

    if len(target_idx) > 0:
        for target_i, target_gallery_idx in enumerate(target_idx):
            target_row = target_i // ncols
            target_col = target_i % ncols
            subplot_idx = (target_row + 1) * ncols + target_col + 1
            ax = plt.subplot(total_rows, ncols, subplot_idx)
            ax.axis('off')
            if first_target_ax is None:
                first_target_ax = ax
            img_path, _ = image_datasets['gallery'].imgs[int(target_gallery_idx)]
            imshow(img_path)
            label = int(gallery_label[int(target_gallery_idx)])
            ax.text(0.5, -0.08, f'ID: {label}', transform=ax.transAxes, ha='center', va='top', color='green')
        for empty_slot in range(len(target_idx), target_rows * ncols):
            target_row = empty_slot // ncols
            target_col = empty_slot % ncols
            subplot_idx = (target_row + 1) * ncols + target_col + 1
            ax = plt.subplot(total_rows, ncols, subplot_idx)
            ax.axis('off')
    else:
        ax = plt.subplot(total_rows, ncols, ncols + 1)
        ax.axis('off')
        first_target_ax = ax
        ax.text(0.5, 0.5, 'No target in gallery (different camera)', transform=ax.transAxes, ha='center', va='center')
        for empty_col in range(1, ncols):
            ax = plt.subplot(total_rows, ncols, ncols + empty_col + 1)
            ax.axis('off')

    title_y = (query_ax.get_position().y0 + first_target_ax.get_position().y1) / 2.0
    fig.text(
        0.5,
        title_y,
        'Target images',
        ha='center',
        va='center',
        fontsize=10,
        color='black',
    )
except RuntimeError:
    for rank_i in range(topk_count):
        img_path = image_datasets['gallery'].imgs[index[rank_i]]
        print(img_path[0])
    print("target_imgs:")
    for target_gallery_idx in target_idx:
        img_path = image_datasets['gallery'].imgs[int(target_gallery_idx)]
        print(img_path[0])
    print('If you want to see the visualization of the ranking result, graphical user interface is needed.')

# Save the figure
fig.savefig(output_filename, bbox_inches='tight', pad_inches=0.03)
print(f"saved_figure: {output_filename}")
