# Meeting Notes

**03/07**  
Project Ideas
* Apply different parallelism techniques for LoRA (pipeline-parallelism, mixture of experts)

Experiments we can do
* Measure latency when we scale (from 1 to multiple nodes, GPUs and cores)
* Measuring latency for different batch size on different setups
* Measuring time to reach accuracy for different ranks

Applications
* Federated learning: Train LoRA's on private data and merge them centrally.  
* INFaaS-like inference with LoRA. Different LoRAs for different user requirements.
