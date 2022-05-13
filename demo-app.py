import streamlit as st
import numpy as np
import sys
import tensorflow as tf
import urllib
import os
import pickle
import PIL
import torch
import matplotlib.pyplot as plt

from SDEdit.to_make_mask import extract_mask,imshow,img2tensor
from SDEdit.SDEdit.main import *
from runners.image_editing import Diffusion


IMG_PATH = '/opt/ml/streamlit-test/runs/image_samples/images'



def main():
    st.title("Let's decorate your ROOM :)")
    st.write("Choose your room image ~")

    c1, c2 = st.columns([1,1])
    with c1:
        original_file = st.file_uploader("Original")
        if original_file:
            st.image(original_file)
    
    with c2:
        sketch_file = st.file_uploader("Sketch")
        if sketch_file:
            st.image(sketch_file)

    custom_mask = extract_mask(input_original = original_file, 
                                input_sketch = sketch_file)

    custom_mask = custom_mask.unsqueeze(dim=0)
    custom_mask = custom_mask.repeat(4, 1, 1, 1)
    x3 = custom_mask
    x3 = (x3 - 0.5) * 2.
    x3 = x3.to("cpu")
    x3 = x3.permute(1, 2, 0, 3)
    x3 = x3.reshape(x3.shape[0], x3.shape[1], -1)
    x3 = x3 / 2 + 0.5     # unnormalize
    x3 = torch.clamp(x3, min=0., max=1.)
    npimg = x3.numpy()
    npimg = np.transpose(npimg, (1, 2, 0))

    # mask 출력
    st.image(npimg)

    # [mask, image] pth로 저장
    #torch.save()


    # 버튼
    original = PIL.Image.open(IMG_PATH + '/original_input.png')
    if st.button('짜잔~'):
        st.write('얍!')
        st.image(original)
        for i in range(3):
            make_image = PIL.Image.open(IMG_PATH + f'/samples_{i}.png')
            st.image(make_image)


if __name__ == "__main__":
    main()