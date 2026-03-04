# Tutorial for TinyChat: Optimizing LLM on Edge Devices

This is a lab for [efficientml.ai course](https://efficientml.ai/).

Running large language models (LLMs) on the edge is of great importance. By embedding LLMs directly into real-world systems such as in-car entertainment systems or spaceship control interfaces, users can access instant responses and services without relying on a stable internet connection. Moreover, this approach alleviates the inconvenience of queuing delays often associated with cloud services. As such, running LLMs on the edge not only enhances user experience but also addresses privacy concerns, as sensitive data remains localized and reduces the risk of potential breaches.

However, despite their impressive capabilities, LLMs have traditionally been quite resource-intensive. They require considerable computational power and memory resources, which makes it challenging to run these models on edge devices with limited capabilities.

In this lab, you will learn the following:
* How to deploy an LLaMA2-7B-chat with TinyChatEngine on your computer.
* Implement different optimization techniques (loop unrolling, multithreading, and SIMD programming) for the linear kernel.
* Observe the end-to-end latency improvement achieved by each technique.


## TinyChatEngine

This tutorial is based on [TinyChatEngine](https://github.com/mit-han-lab/TinyChatEngine), a powerful neural network library specifically designed for the efficient deployment of quantized large language models (LLMs) on edge devices. 

![demo](assets/figures/chat.gif)

## Tutorial document

Please check this document and follow the instructions which will walk you through the tutorial: https://docs.google.com/document/d/13IaTfPKjp0KiSBEhPdX9IxgXMIAZfiFjor37OWQJhMM/edit?usp=sharing

## Submission

* Report: Please write a report ([form](https://docs.google.com/document/d/17Z_ab8EhDvjcigLXdDqMqd2LTVsZ4CnpOYNkRTrnTmU/edit?usp=sharing)) that includes your code and the performance improvement for each starter code. 
* Code: Use `git diff` to generate a patch for your implementation. We will use this patch to test the correctness of your code. Please name your patch as `{studentID}-{ISA}.patch` where {ISA} should be one of x86 and ARM, depending on your computer.

## Related Projects

[TinyChatEngine](https://github.com/mit-han-lab/TinyChatEngine).

[TinyEngine](https://github.com/mit-han-lab/tinyengine).

[Smoothquant](https://github.com/mit-han-lab/smoothquant).

[AWQ: Activation-aware Weight Quantization for LLM Compression and Acceleration](https://github.com/mit-han-lab/llm-awq)

## Acknowledgement

[llama.cpp](https://github.com/ggerganov/llama.cpp)

[transformers](https://github.com/huggingface/transformers)

# 我们在此Lab中的目的

我们的目的是完成 `kernel/starter_code/` 下的代码，因为此代码会被用在 模型推理时的 `transformer/src/ops/linear.cc` 下的 `void Linear_FP_int4::forward_ref(const Matrix3D<float> &a, Matrix3D<float> &c)` 函数，依据我们编译时传入的参数 IMP

如果想要尝试使用chat, 则可以使用命令`make chat -j IMP=0` 来编译

可以通过`make --dry-run --always-make chat -j IMP=0` 查看编译过程

我使用`python download_model.py --model LLaMA_7B_2_chat --QM QM_x86` 下载模型，然后运行了他，发现这个模型是个"智障"...
