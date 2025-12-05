# All Gather Microbenchmark

This guide demonstrates how to performance collectives benchmark for the `2x2x1` and `2x2x2` topologies.

## Setup
Please follow the prerequisites setup [here](../../Ironwood_Microbenchmarks_readme.md#prerequisites).

### Create Node Pool

Create a `2x2x2` node pool in your GKE cluster. For the `2x2x1` topology, please follow the instructions provided [here](../../Ironwood_Microbenchmarks_readme.md#setup).

```bash
gcloud compute resource-policies create workload-policy ${WORKLOAD_POLICY_NAME} \
    --type HIGH_THROUGHPUT \
    --accelerator-topology 2x2x2 \
    --project ${PROJECT_ID} \
    --region ${REGION}

gcloud container node-pools create ${TPU7X_2X2X2_NODE_POOL_NAME} \
    --cluster=${CLUSTER_NAME} \
    --machine-type=tpu7x-standard-4t \
    --placement-policy=${WORKLOAD_POLICY_NAME} \
    --project ${PROJECT_ID} \
    --location=${LOCATION} \
    --reservation=${RESERVATION_NAME} \
    --reservation-affinity=specific

```

## Run Benchmarks

Deploy the all_gather microbenchmarks for different TPU topologies.
```
kubectl apply -f tpu7x-2x2x1-ici-all-gather-microbenchmark.yaml
kubectl apply -f tpu7x-2x2x2-ici-all-gather-microbenchmark.yaml
```

### Monitor Results

Use the following commands to retrieve the benchmark logs.
```bash
kubectl logs pod/tpu7x-2x2x1-ici-all-gather-microbenchmark
kubectl logs job.batch/tpu7x-2x2x2-ici-all-gather-microbenchmark
```

Once the benchmark completes, you should see logs similar to the example below:
```bash
metadata:  {'iteration': 16384, 'op_type': 'AG' ... }
metrics:  {... 'achieved_bw (GB/s)_max': np.float64(159.35667369827922) ...}
Writing metrics to JSONL file: ../microbenchmarks/all_gather/metrics_report.jsonl
Metrics written to CSV at ../microbenchmarks/all_gather/t_all_gather_XC9DQ60YW6.tsv.
```

To retrieve the complete results, use the `kubectl cp` command to copy the TSV report file, typically named `t_all_gather_xxxx.tsv`, from the `/microbenchmarks` directory within the pod.

### Cleanup

```bash
kubectl delete -f tpu7x-2x2x1-ici-all-gather-microbenchmark.yaml
kubectl delete -f tpu7x-2x2x2-ici-all-gather-microbenchmark.yaml
```

## Expected Results
| Topology | Number of Elements | Achieved Bandwidth (GB/s) | Transferred Data (GB) | Input Shape      | Output  Shape    |
| -------- | ------------------ | ------------------------- | --------------------- | ---------------- | ---------------- |
| 2x2x1    | 64                 | 71.7703034                | 0.001572864           | f32[64,8,128]    | f32[256,8,128]   |
| 2x2x1    | 256                | 129.1869148               | 0.006291456           | f32[256,8,128]   | f32[1024,8,128]  |
| 2x2x1    | 1024               | 161.9040742               | 0.025165824           | f32[1024,8,128]  | f32[4096,8,128]  |
| 2x2x1    | 4096               | 174.765465                | 0.100663296           | f32[4096,8,128]  | f32[16384,8,128] |
| 2x2x1    | 16384              | 178.7714158               | 0.402653184           | f32[16384,8,128] | f32[65536,8,128] |
| 2x2x2    | 64                 | 69.53210305               | 0.001572864           | f32[64,8,128]    | f32[256,8,128]   |
| 2x2x2    | 256                | 127.3945462               | 0.006291456           | f32[256,8,128]   | f32[1024,8,128]  |
| 2x2x2    | 1024               | 162.0864029               | 0.025165824           | f32[1024,8,128]  | f32[4096,8,128]  |
| 2x2x2    | 4096               | 174.6652133               | 0.100663296           | f32[4096,8,128]  | f32[16384,8,128] |
| 2x2x2    | 16384              | 178.7592252               | 0.402653184           | f32[16384,8,128] | f32[65536,8,128] |
