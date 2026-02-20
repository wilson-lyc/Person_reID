import scipy.io
import torch
import numpy as np
#import time
import os
from lark import send_message, log_info
from datetime import datetime

run_id = datetime.now().strftime("%Y%m%d_%H%M%S")

#######################################################################
# Evaluate
def evaluate(qf,ql,qc,gf,gl,gc):
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
    # mask = np.in1d(index, junk_index, invert=True) # old numpy
    mask = np.isin(index, junk_index, invert=True) # new numpy
    index = index[mask]

    # find good_index index
    ngood = len(good_index)
    # mask = np.in1d(index, good_index) # old numpy
    mask = np.isin(index, good_index) # new numpy
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
result = scipy.io.loadmat('pytorch_result.mat')
query_feature = torch.FloatTensor(result['query_f'])
query_cam = result['query_cam'][0]
query_label = result['query_label'][0]
gallery_feature = torch.FloatTensor(result['gallery_f'])
gallery_cam = result['gallery_cam'][0]
gallery_label = result['gallery_label'][0]

multi = os.path.isfile('multi_query.mat')

if multi:
    m_result = scipy.io.loadmat('multi_query.mat')
    mquery_feature = torch.FloatTensor(m_result['mquery_f'])
    mquery_cam = m_result['mquery_cam'][0]
    mquery_label = m_result['mquery_label'][0]
    mquery_feature = mquery_feature.cuda()

query_feature = query_feature.cuda()
gallery_feature = gallery_feature.cuda()

print(query_feature.shape)
CMC = torch.IntTensor(len(gallery_label)).zero_()
ap = 0.0
#print(query_label)
for i in range(len(query_label)):
    ap_tmp, CMC_tmp = evaluate(query_feature[i],query_label[i],query_cam[i],gallery_feature,gallery_label,gallery_cam)
    if CMC_tmp[0]==-1:
        continue
    CMC = CMC + CMC_tmp
    ap += ap_tmp
    #print(i, CMC_tmp[0])

CMC = CMC.float()
CMC = CMC/len(query_label) #average CMC
rank1 = CMC[0].item()
rank5 = CMC[4].item()
rank10 = CMC[9].item()
mAP = (ap/len(query_label)).item()
print('Rank@1:%f Rank@5:%f Rank@10:%f mAP:%f'%(rank1, rank5, rank10, mAP))

# 发送评估结果消息
send_message("评估完成", f"Run ID: {run_id}\nRank@1: {rank1:.4f}\nmAP: {mAP:.4f}")

# 记录评估日志
log_info("evaluate_gpu.py", run_id, {
    "rank1": f"{rank1:.4f}",
    "rank5": f"{rank5:.4f}",
    "rank10": f"{rank10:.4f}",
    "mAP": f"{mAP:.4f}"
})

# multiple-query
CMC = torch.IntTensor(len(gallery_label)).zero_()
ap = 0.0
if multi:
    for i in range(len(query_label)):
        mquery_index1 = np.argwhere(mquery_label==query_label[i])
        mquery_index2 = np.argwhere(mquery_cam==query_cam[i])
        mquery_index =  np.intersect1d(mquery_index1, mquery_index2)
        mq = torch.mean(mquery_feature[mquery_index,:], dim=0)
        ap_tmp, CMC_tmp = evaluate(mq,query_label[i],query_cam[i],gallery_feature,gallery_label,gallery_cam)
        if CMC_tmp[0]==-1:
            continue
        CMC = CMC + CMC_tmp
        ap += ap_tmp
        #print(i, CMC_tmp[0])
    CMC = CMC.float()
    CMC = CMC/len(query_label) #average CMC
    multi_rank1 = CMC[0].item()
    multi_rank5 = CMC[4].item()
    multi_rank10 = CMC[9].item()
    multi_mAP = (ap/len(query_label)).item()
    print('multi Rank@1:%f Rank@5:%f Rank@10:%f mAP:%f'%(multi_rank1, multi_rank5, multi_rank10, multi_mAP))

    # 发送多查询评估结果消息
    send_message("多查询评估完成", f"Run ID: {run_id}\nRank@1: {multi_rank1:.4f}\nmAP: {multi_mAP:.4f}")
