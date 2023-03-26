## Description:

It is the implementation of the paper: ImageNet-trained CNNs are biased towards texture; increasing shape bias improves accuracy and robustness

## Instructions:
1. To run the style transferred images of the paper:

```bash
python inference.py
```

2. To perform your own style transfer:

    2.1 Download the style transfer dataset: https://www.kaggle.com/c/painter-by-numbers/data
   
    2.2 Put style transfer data and COCO dataset at the prescribed path in ```configs.py```

```bash
python style_transfer.py
python inference.py
```
