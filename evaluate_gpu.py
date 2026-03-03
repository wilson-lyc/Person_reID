import scipy.io
import torch
import numpy as np
import time
import os
import argparse
from tool.lark import lark_log, lark_notify
from tool.run_id import generate_run_id
from tool.eval_result_format import (
    append_result_txt,
    build_result_block,
    format_elapsed,
    format_torch_size,
    infer_eval_dataset,
    infer_model_name,
    now_str,
)

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
parser = argparse.ArgumentParser(description='Evaluate (GPU)')
parser.add_argument('--result_mat', default='pytorch_result.mat', type=str, help='path to pytorch result mat')
parser.add_argument('--multi_mat', default='', type=str, help='path to multi-query mat (optional)')
parser.add_argument('--run_id', default='', type=str, help='external run id for logging')
args = parser.parse_args()
run_id = args.run_id.strip() or generate_run_id()

result = scipy.io.loadmat(args.result_mat)
since = time.time()
query_feature = torch.FloatTensor(result['query_f'])
query_cam = result['query_cam'][0]
query_label = result['query_label'][0]
gallery_feature = torch.FloatTensor(result['gallery_f'])
gallery_cam = result['gallery_cam'][0]
gallery_label = result['gallery_label'][0]

multi_mat_path = args.multi_mat.strip()
if multi_mat_path:
    multi = os.path.isfile(multi_mat_path)
else:
    fallback_multi = os.path.join(os.path.dirname(args.result_mat) or '.', 'multi_query.mat')
    if not os.path.isfile(fallback_multi):
        dataset_suffix = os.path.basename(args.result_mat).replace('pytorch_result_', '')
        fallback_multi = os.path.join(os.path.dirname(args.result_mat) or '.', f'multi_query_{dataset_suffix}')
    multi = os.path.isfile(fallback_multi)
    multi_mat_path = fallback_multi

if multi:
    m_result = scipy.io.loadmat(multi_mat_path)
    mquery_feature = torch.FloatTensor(m_result['mquery_f'])
    mquery_cam = m_result['mquery_cam'][0]
    mquery_label = m_result['mquery_label'][0]
    mquery_feature = mquery_feature.cuda()

query_feature = query_feature.cuda()
gallery_feature = gallery_feature.cuda()

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
rank1 = float(CMC[0].item())
rank5 = float(CMC[4].item())
rank10 = float(CMC[9].item())
map_score = float(ap/len(query_label))

# multiple-query
CMC = torch.IntTensor(len(gallery_label)).zero_()
ap = 0.0
multi_rank1 = None
multi_rank5 = None
multi_rank10 = None
multi_map = None
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
    multi_rank1 = float(CMC[0].item())
    multi_rank5 = float(CMC[4].item())
    multi_rank10 = float(CMC[9].item())
    multi_map = float(ap/len(query_label))

time_elapsed = time.time() - since
evaluated_at = now_str()
model_name = infer_model_name(args.result_mat)
eval_dataset = infer_eval_dataset(args.result_mat)
elapsed_text = format_elapsed(time_elapsed)
extra_metric_lines = []
if multi and multi_rank1 is not None:
    extra_metric_lines.append(
        f"multi Rank@1:{multi_rank1:.6f} Rank@5:{multi_rank5:.6f} "
        f"Rank@10:{multi_rank10:.6f} mAP:{multi_map:.6f}"
    )
result_block = build_result_block(
    evaluated_at=evaluated_at,
    model_name=model_name,
    eval_dataset=eval_dataset,
    result_mat=os.path.basename(args.result_mat),
    elapsed_text=elapsed_text,
    feature_shape_text=format_torch_size(query_feature.shape),
    rank1=rank1,
    rank5=rank5,
    rank10=rank10,
    map_score=map_score,
    extra_metric_lines=extra_metric_lines,
)
print(result_block)
result_txt_path = append_result_txt(args.result_mat, result_block)

lark_log(
    project="Person_reID",
    file="evaluate_gpu.py",
    run_id=run_id,
    log={
        "event": "evaluate_end",
        "result_mat": args.result_mat,
        "multi_mat": multi_mat_path if multi else "",
        "multi_used": bool(multi),
        "rank1": rank1,
        "rank5": rank5,
        "rank10": rank10,
        "mAP": map_score,
        "multi_rank1": multi_rank1,
        "multi_rank5": multi_rank5,
        "multi_rank10": multi_rank10,
        "multi_mAP": multi_map,
        "elapsed_seconds": float(time_elapsed),
        "result_txt": result_txt_path,
    },
)
lark_notify(
    title="[Evaluate End] evaluate_gpu.py",
    msg=(
        f"run_id={run_id}\n"
        f"model_name={model_name}, eval_dataset={eval_dataset}\n"
        f"result_mat={os.path.basename(args.result_mat)}\n"
        f"Rank@1={rank1:.6f}, Rank@5={rank5:.6f}, Rank@10={rank10:.6f}, mAP={map_score:.6f}\n"
        f"multi_used={bool(multi)}"
        + (
            f"\nmulti Rank@1={multi_rank1:.6f}, Rank@5={multi_rank5:.6f}, "
            f"Rank@10={multi_rank10:.6f}, mAP={multi_map:.6f}"
            if multi and multi_rank1 is not None else ""
        )
        + f"\nelapsed={elapsed_text}\nresult_txt={result_txt_path}"
    ),
)
