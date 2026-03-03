import scipy.io
import torch
import numpy as np
import time
import argparse
import os
from  re_ranking import re_ranking
from tool.eval_result_format import (
    append_result_txt,
    build_result_block,
    format_elapsed,
    infer_eval_dataset,
    infer_model_name,
    now_str,
)
#######################################################################
# Evaluate
def evaluate(score,ql,qc,gl,gc):
    index = np.argsort(score)  #from small to large
    #index = index[::-1]
    # good index
    query_index = np.argwhere(gl==ql)
    camera_index = np.argwhere(gc==qc)

    good_index = np.setdiff1d(query_index, camera_index, assume_unique=True)
    junk_index1 = np.argwhere(gl==-1)
    junk_index2 = np.intersect1d(query_index, camera_index)
    junk_index = np.append(junk_index2, junk_index1) #.flatten())
    
    CMC_tmp = compute_mAP(index, good_index, junk_index)
    return CMC_tmp


def compute_mAP(index, good_index, junk_index):
    ap = 0
    cmc = torch.IntTensor(len(index)).zero_()
    if good_index.size==0:   # if empty
        cmc[0] = -1
        return ap,cmc

    # remove junk_index
    mask = np.in1d(index, junk_index, invert=True)
    index = index[mask]

    # find good_index index
    ngood = len(good_index)
    mask = np.in1d(index, good_index)
    rows_good = np.argwhere(mask==True)
    rows_good = rows_good.flatten()
    
    cmc[rows_good[0]:] = 1
    for i in range(ngood):
        d_recall = 1.0/ngood
        precision = (i+1)*1.0/(rows_good[i]+1)
        if rows_good[i]!=0:
            old_precision = i*1.0/rows_good[i]
        else:
            old_precision=1.0
        ap = ap + d_recall*(old_precision + precision)/2

    return ap, cmc

######################################################################
parser = argparse.ArgumentParser(description='Evaluate Re-ranking')
parser.add_argument('--result_mat', default='pytorch_result.mat', type=str, help='path to pytorch result mat')
args = parser.parse_args()

result = scipy.io.loadmat(args.result_mat)
query_feature = result['query_f']
query_cam = result['query_cam'][0]
query_label = result['query_label'][0]
gallery_feature = result['gallery_f']
gallery_cam = result['gallery_cam'][0]
gallery_label = result['gallery_label'][0]

CMC = torch.IntTensor(len(gallery_label)).zero_()
ap = 0.0
#re-ranking
q_g_dist = np.dot(query_feature, np.transpose(gallery_feature))
q_q_dist = np.dot(query_feature, np.transpose(query_feature))
g_g_dist = np.dot(gallery_feature, np.transpose(gallery_feature))
since = time.time()
re_rank = re_ranking(q_g_dist, q_q_dist, g_g_dist)
time_elapsed = time.time() - since
for i in range(len(query_label)):
    ap_tmp, CMC_tmp = evaluate(re_rank[i,:],query_label[i],query_cam[i],gallery_label,gallery_cam)
    if CMC_tmp[0]==-1:
        continue
    CMC = CMC + CMC_tmp
    ap += ap_tmp
    #print(i, CMC_tmp[0])

CMC = CMC.float()
CMC = CMC/len(query_label) #average CMC
rank1 = float(CMC[0].item())
rank5 = float(CMC[4].item())
rank10 = float(CMC[9].item())
map_score = float(ap/len(query_label))
evaluated_at = now_str()
model_name = infer_model_name(args.result_mat)
eval_dataset = infer_eval_dataset(args.result_mat)
elapsed_text = format_elapsed(time_elapsed)
result_block = build_result_block(
    evaluated_at=evaluated_at,
    model_name=model_name,
    eval_dataset=eval_dataset,
    result_mat=os.path.basename(args.result_mat),
    elapsed_text=elapsed_text,
    rank1=rank1,
    rank5=rank5,
    rank10=rank10,
    map_score=map_score,
    extra_metric_lines=[
        f"rerank Rank@1:{rank1:.6f} Rank@5:{rank5:.6f} Rank@10:{rank10:.6f} mAP:{map_score:.6f}"
    ],
)
print(result_block)
append_result_txt(args.result_mat, result_block)
