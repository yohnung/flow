export LD_LIBRARY_PATH=/search/odin/tools/cuda-10.0/lib64

  

# for train
#CUDA_VISIBLE_DEVICES="6" python run_flow.py \
#    --data_dir=/search/odin/chenwm/results_arxived/hnsw/compare_new_cosine_albert_and_pca/bilinear-albert_ctrst/pairwise_ctrst_2/run_flow/embeddings.shuf.dat \
#    --do_train=True

# for eval
#for title
sed -i 's/line_seg\[1\]/line_seg\[7\]/' run_flow.py
CUDA_VISIBLE_DEVICES="6" python run_flow.py \
    --data_dir=/search/odin/chenwm/results_arxived/hnsw/compare_new_cosine_albert_and_pca/bilinear-albert_ctrst/pairwise_ctrst_2/run_flow/embeddings.shuf.dat \
    --do_train=False \
    --do_predict=True \
    --test_file=/search/odin/chenwm/results_arxived/hnsw/compare_new_cosine_albert_and_pca/bilinear-albert_ctrst/pairwise_ctrst_2/build_and_query_togather/title-docid.shuf.emb.dat \
    --init_checkpoint=/search/odin/chenwm/flow/results_train_2000/model.ckpt-2000
sed -i 's/line_seg\[7\]/line_seg\[1\]/' run_flow.py

# for query
CUDA_VISIBLE_DEVICES="6" python run_flow.py \
    --data_dir=/search/odin/chenwm/results_arxived/hnsw/compare_new_cosine_albert_and_pca/bilinear-albert_ctrst/pairwise_ctrst_2/run_flow/embeddings.shuf.dat \
    --do_train=False \
    --do_predict=True \
    --test_file=/search/odin/chenwm/results_arxived/hnsw/compare_new_cosine_albert_and_pca/bilinear-albert_ctrst/pairwise_ctrst_2/build_and_query_togather/segmented.query.emb.dat \
    --init_checkpoint=/search/odin/chenwm/flow/results_train_2000/model.ckpt-2000
