Face Detection and Recognition Project.**

```bash
# Face detection on validation data
python task1.py --input_path validation_folder/images --output ./result_task1_val.json

# Validation
python ComputeFBeta/ComputeFBeta.py --preds result_task1_val.json --groundtruth validation_folder/ground-truth.json
```

```bash
# Face detection on test data
python task1.py --input_path test_folder/images --output ./result_task1.json
```

**task 2 running**
```bash
python task2.py --input_path faceCluster_5 --num_cluster 5
```
