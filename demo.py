import argparse
import scipy.io
import torch
import numpy as np
import os
from torchvision import datasets
import matplotlib
matplotlib.use('agg')
import matplotlib.pyplot as plt

###############################################################################
# Visualization style (clean scientific style)
COLOR_BG = '#FFFFFF'
COLOR_TEXT = '#1F2937'
COLOR_HIT = '#1F8A4C'
COLOR_MISS = '#C83D3D'
COLOR_QUERY = '#2F6DB3'
COLOR_BORDER = '#D5DBE3'
COLOR_STRIP_BG = '#F5F7FA'

FONT_TITLE = 16
FONT_SUBTITLE = 11
FONT_CARD_TITLE = 10
FONT_CAPTION = 10

CARD_BORDER_WIDTH = 2.0
CARD_BORDER_WIDTH_LIGHT = 1.2
TOPK = 10
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
    ax = plt.gca()
    ax.imshow(im)
    if title is not None:
        ax.set_title(title, fontsize=FONT_CARD_TITLE, color=COLOR_TEXT)
    plt.pause(0.001)  # pause a bit so that plots are updated


def style_axis(ax, edge_color, edge_width):
    """Apply a unified card style to an axis."""
    ax.set_xticks([])
    ax.set_yticks([])
    ax.tick_params(left=False, bottom=False, labelleft=False, labelbottom=False)
    for spine in ax.spines.values():
        spine.set_visible(True)
        spine.set_linewidth(edge_width)
        spine.set_edgecolor(edge_color)
    ax.set_facecolor(COLOR_BG)


def add_caption(ax, text, color=COLOR_TEXT, size=FONT_CAPTION):
    """Add centered caption inside image card to avoid clipping."""
    ax.text(
        0.5,
        0.03,
        text,
        transform=ax.transAxes,
        ha='center',
        va='bottom',
        fontsize=size,
        color=color,
        bbox=dict(facecolor='white', edgecolor='none', alpha=0.72, boxstyle='round,pad=0.2'),
    )

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
grid_rows = 3 + target_rows
topk_hits = int(np.sum(gallery_label[index[:topk_count]] == query_pid))
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
fig = plt.figure(figsize=(max(14.2, ncols * 1.18), 2.85 + target_rows * 2.20), facecolor=COLOR_BG)
fig.subplots_adjust(left=0.018, right=0.988, top=0.95, bottom=0.04, wspace=0.045, hspace=0.08)
grid = fig.add_gridspec(grid_rows, ncols, height_ratios=[0.20, 1.0, 0.20] + [1.0] * target_rows)
fig.suptitle('Person ReID Demo', fontsize=FONT_TITLE, color=COLOR_TEXT, y=0.965)
summary_line = (
    f'Query ID: {query_pid} | '
    f'Query Cam: {query_camera} | '
    f'Target Count: {len(target_idx)} | Top{topk_count} Hits: {topk_hits}'
)
try: # Visualize Ranking Result 
    # Graphical User Interface is needed
    summary_ax = fig.add_subplot(grid[0, :])
    summary_ax.axis('off')
    summary_ax.text(
        0.5,
        0.5,
        summary_line,
        ha='center',
        va='center',
        fontsize=FONT_SUBTITLE,
        color=COLOR_TEXT,
    )

    query_ax = fig.add_subplot(grid[1, 0])
    imshow(query_path)
    style_axis(query_ax, COLOR_QUERY, CARD_BORDER_WIDTH)
    query_ax.set_title('Query', fontsize=FONT_CARD_TITLE, color=COLOR_QUERY, pad=4)
    add_caption(query_ax, f'ID:{query_pid}  CAM:{query_camera}', color=COLOR_TEXT)
    for rank_i in range(topk_count):
        ax = fig.add_subplot(grid[1, rank_i + 1])
        img_path, _ = image_datasets['gallery'].imgs[index[rank_i]]
        label = int(gallery_label[index[rank_i]])
        imshow(img_path)
        matched = label == query_pid
        rank_color = COLOR_HIT if matched else COLOR_MISS
        style_axis(ax, rank_color, CARD_BORDER_WIDTH)
        ax.set_title(f'Rank {rank_i + 1}', fontsize=FONT_CARD_TITLE, color=rank_color, pad=4)
        add_caption(ax, f'ID:{label}', color=rank_color)
        print(img_path)
    for empty_col in range(topk_count + 1, ncols):
        ax = fig.add_subplot(grid[1, empty_col])
        style_axis(ax, COLOR_BORDER, CARD_BORDER_WIDTH_LIGHT)
        ax.axis('off')

    strip_ax = fig.add_subplot(grid[2, :])
    strip_ax.set_facecolor(COLOR_STRIP_BG)
    style_axis(strip_ax, COLOR_BORDER, CARD_BORDER_WIDTH_LIGHT)
    strip_ax.text(
        0.5,
        0.5,
        f'Target Images - {len(target_idx)}',
        ha='center',
        va='center',
        fontsize=FONT_CARD_TITLE,
        color=COLOR_TEXT,
    )

    if len(target_idx) > 0:
        for target_i, target_gallery_idx in enumerate(target_idx):
            target_row = target_i // ncols
            target_col = target_i % ncols
            ax = fig.add_subplot(grid[target_row + 3, target_col])
            img_path, _ = image_datasets['gallery'].imgs[int(target_gallery_idx)]
            imshow(img_path)
            style_axis(ax, COLOR_BORDER, CARD_BORDER_WIDTH_LIGHT)
        for empty_slot in range(len(target_idx), target_rows * ncols):
            target_row = empty_slot // ncols
            target_col = empty_slot % ncols
            ax = fig.add_subplot(grid[target_row + 3, target_col])
            style_axis(ax, COLOR_BORDER, CARD_BORDER_WIDTH_LIGHT)
            ax.axis('off')
    else:
        ax = fig.add_subplot(grid[3, 0])
        style_axis(ax, COLOR_BORDER, CARD_BORDER_WIDTH_LIGHT)
        ax.set_facecolor(COLOR_STRIP_BG)
        ax.text(
            0.5,
            0.5,
            'No target in gallery (different camera)',
            transform=ax.transAxes,
            ha='center',
            va='center',
            fontsize=FONT_CARD_TITLE,
            color=COLOR_TEXT,
        )
        for empty_col in range(1, ncols):
            ax = fig.add_subplot(grid[3, empty_col])
            style_axis(ax, COLOR_BORDER, CARD_BORDER_WIDTH_LIGHT)
            ax.axis('off')
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
