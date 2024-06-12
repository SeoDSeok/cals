This is the project code that proceeded to extract text from the image data related to the design budget and design report through OCR at the construction site in Korea.
The .h5 file of the MRCNN model used in connection is not uploaded due to its large capacity.

_Before publishing, please either delete this file, or edit it to describe your project._

### Environment ( Docker image ) 
```md
docker pull tensorflow/tensorflow:2.7.0-gpu
```

### Download Mask-Rcnn Model (Figure & Formular Detection)

Already Uploaded Mask-RCNN model to Git. 
If you have problem to download model, you can download with below code 
```md 
wget --load-cookies ~/cookies.txt "https://docs.google.com/uc?export=download&confirm=$(wget --quiet --save-cookies ~/cookies.txt --keep-session-cookies --no-check-certificate 'https://drive.google.com/file/d/1PTzFMJp-pF2Tt-EwPyibfj2w0KMfm9Mi/view?usp=sharing' -O- | sed -rn 's/.*confirm=([0-9A-Za-z_]+).*/\1\n/p')&id=1PTzFMJp-pF2Tt-EwPyibfj2w0KMfm9Mi" -O capstone_200_ppt.h5 && rm -rf ~/cookies.txt
```
### Activate Our Solution 

```md 
python main.py [file_path] [file_type] [api_key] [url]
```

## Mask-RCNN Custom 

### code 
```md
git lfs pull
```
