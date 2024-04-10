## Dataset information

Images used for training YOLO for object detection:
 - first training instance: 697
 - second training instance: 1296

<!-- Steps to serve local files over url. -->


# generate a manifest
```
./scripts/generate_folder_manifest.sh \
    /cs-share/pradalier/tmp/judo \
    '*/referee_v2/*.png' \
    /cs-share/pradalier/tmp/judo/referee_files.txt
```
# start up nginx
```
./scripts/serve_local_files.sh \
    /cs-share/pradalier/tmp/judo

```


<!-- Setting-up label-studio backend -->
## pre-annotation and active labeling

```bash
python -m user.tsutar.label_studio_backend.referee_pose.wsgi \
    --model-dir /tmp/model \
    --debug \
    --api-token=...

python -m user.tsutar.label_studio_backend.referee_pose.wsgi \
    --model-dir /tmp/model \
    --model-name /cs-share/pradalier/tmp/judo/models/referee_pose/v1/weights/03-04-2024-referee-best.pt\
    --debug \
    --api-token=...
```
