import torch

def numerical_test(tensors1, tensors2, rtol, atol):
    """
    truth_tensors is the source of truth.
    test_dict looks like
    [
        (name, out_tensors, atol, rtol),
        ...
    ]
    """
    assert len(tensors1) == len(tensors2)
    n_failures = 0
    max_diff = 0
    for tensor1, tensor2 in zip(tensors1, tensors2):
        max_diff = max(max_diff, torch.max(torch.abs(tensor1 - tensor2)))
        if not torch.allclose(tensor1, tensor2, atol=atol, rtol=rtol):
            n_failures += 1

    if n_failures == 0:
        print(f"Numerical test PASS")
    else:
        print(f"Numerical test FAIL {n_failures}/{len(tensors1)}. Max diff is {max_diff}")

def benchmark_torch_function(is_cuda, f, inputs):
    """
    Benchmark torch on gpu or cpu
    """
    iters = len(inputs)
    if(is_cuda):
        f(inputs[0])
        torch.cuda.synchronize()
        start_event = torch.cuda.Event(enable_timing=True)
        end_event = torch.cuda.Event(enable_timing=True)
        start_event.record()
        for x in inputs:
            f(x)
        end_event.record()
        torch.cuda.synchronize()
        return (start_event.elapsed_time(end_event) * 1.0e-3) / iters
    else:
        import time

        f(inputs[0])
        start_event = time.perf_counter()
        for x in inputs:
            f(x)
        end_event = time.perf_counter()
        return (end_event - start_event) / iters

def get_outputs(f, inputs):
    outputs = [f(x).logits for x in inputs]
    return outputs


def benchmark():
    from transformers import AutoModelForSequenceClassification
    
    num_batches = 500
    batch_size = 1
    seqlen = 25
    is_cuda = True # TODO CPU + half does not work. fp32 option?
    device = 'cuda' if is_cuda else 'cpu' 

    model = AutoModelForSequenceClassification.from_pretrained("bert-base-cased").eval().to(device).half()

     # batch size 1, seqlen 25
    eval_inputs = torch.randint(low=1, high=25000, size=(num_batches, batch_size, seqlen)).to(device)

    with torch.no_grad():
        hf_t = benchmark_torch_function(is_cuda, model, eval_inputs)
        out = get_outputs(model, eval_inputs)
    print(f"HF time per batch {hf_t}")

    # right now, need to set all these again because the new module doesn't automatically configure itself
    model.bert.encoder = model.bert.encoder.to_fast()
    model = model.eval().to(device).half()

    with torch.no_grad():
        bt_t = benchmark_torch_function(is_cuda, model, eval_inputs)
        out2 = get_outputs(model, eval_inputs)
    print(f"BT time per batch {bt_t}")

    numerical_test(out, out2, 0, 1e-2) # absolute difference of 1 percent

if __name__=="__main__":
    benchmark()
    
