### Code description
This is the project code that proceeded to extract text from the image data related to the design budget and design report through OCR at the construction site in Korea.

- code/table.py
 : Code for storing design budgets in a database
- code/document.py
 : Code that allows design reports to be stored in a database

### Download Mask-Rcnn Model (Figure & Formular Detection)

Already Uploaded Mask-RCNN model to Git. 
If you have problem to download model, you can download with below code 
```md 
wget --load-cookies ~/cookies.txt "https://docs.google.com/uc?export=download&confirm=$(wget --quiet --save-cookies ~/cookies.txt --keep-session-cookies --no-check-certificate 'https://drive.google.com/file/d/1g8KdJ9PDYQJOzxc2HAJa-SoxwI4RHukB/view?usp=sharing' -O- | sed -rn 's/.*confirm=([0-9A-Za-z_]+).*/\1\n/p')&id=1g8KdJ9PDYQJOzxc2HAJa-SoxwI4RHukB" -O mask_custom.h5 && rm -rf ~/cookies.txt
```
### Activate Our Solution 

```md 
python main.py [file_path] [file_type] [api_key] [url]
```

