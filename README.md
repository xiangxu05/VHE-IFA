# Visualizing the Process of Hieroglyphic Evolution with Inter-Frame Attention

> This study constructed a data set named Oracle to Simplified Chinese (OtSC105) for the evolution of hieroglyphs. Using an inter-frame attention extraction algorithm, images were generated to achieve the visualization of hieroglyphic evolution from oracle bone inscriptions to modern Chinese characters. 


![The Evolution of the Character ”Zao”(meaning early)](./figs/example1.gif)
![The Evolution of the Character ”Qiu”(meaning imprisonment)](./figs/example2.gif)
## Getting Started

We need the following listed environments：
- torch 1.8.0
- python 3.8
- skimage 0.19.2
- numpy 1.23.1
- opencv-python 4.6.0
- timm 0.6.11
- tqdm


## Play with Demos

1. Download the model [checkpoints](https://drive.google.com/drive/folders/1McOO_yt_bPIf0Zk2Ypk9wESXRsXHNaL1?usp=drive_link) and put the ckpt folder into the root dir.
2. Download the dataset [OtSC105](https://drive.google.com/drive/folders/1m4s5Lr2BnVDrhrJg_Pnz24xRn_p4fwcS?usp=drive_link) or your own pictures, then put it into the folder 'Sources'.
3. Run the following commands to generate Nx (arbitrary) frame interpolation demos:

```sh
python Generate_from_dataset.py --model /ours_t/ours --InputPath /Your/Dataset/Path --OutputPath /Your/Output/Path --n /Insert/Frames
```


## Evaluation

1. Using the testbench in the folder of 'Sources'
2. Download the model [checkpoints](https://drive.google.com/drive/folders/1McOO_yt_bPIf0Zk2Ypk9wESXRsXHNaL1?usp=drive_link) and put the ckpt folder into the root dir.
3. For 2x interpolation benchmarks:
   ```sh
   python OtSC105.py --model /ours/ours_small --path /Your/Dataset/Path
   ```

