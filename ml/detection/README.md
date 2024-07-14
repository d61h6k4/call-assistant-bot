
```bash
cd dataset && mkdir tmp && cd tmp
gsutil copy gs://... .
tar xvf ...
ffmpeg -i meeting_record.mp4 -r 0.1 ../images/gnccxwmbdr_fps01_frame_%04d.png
```
