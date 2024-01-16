# judo-footage-analysis

## quickstart

Checkout the repo, and the submodules too:

```bash
git checkout {repo}
git submodule update --init --recursive
```

Install the relevant tools:

- ffmpeg
- b2-tools

For detectron, read through this link:

- https://detectron2.readthedocs.io/en/latest/tutorials/install.html

Install the vendored version, which links to a version of detectron patches with quality-of-life improvements.

## footage links

### 2023 President's Cup

1. https://www.youtube.com/watch?v=uPhXtW_f_AE
2. https://www.youtube.com/watch?v=Nh_cb1RNV9o
3. https://www.youtube.com/watch?v=95paesJw7pk
4. https://www.youtube.com/watch?v=xVDw2bhFXgk
5. https://www.youtube.com/watch?v=2hYzoJ8HSkk
6. https://www.youtube.com/watch?v=B38xef6cHHk
7. https://www.youtube.com/watch?v=mDFtwQVP9GM
8. https://www.youtube.com/watch?v=6rZvqhUaxOE
   - The only footage without smoothcomp overlay for match statistics
9. https://www.youtube.com/watch?v=pE_mDPF0gwI
10. https://www.youtube.com/watch?v=OwhJQFx27YM

## notes

I create a n2d instance with local-ssd (375gb) to download the streamed videos and upload them into a cloud bucket.
I focus on a subset of videos, in particular mat 1 and mat 8.
The former has the [smoothcomp overlay](https://smoothcomp.com/en), which can be used to segment the videos and extract match information from the competition.
The latter does not have an overlay (likely due to user error), and can be used to test algorithms on a raw camera feed.

Install [yt-dlp](https://github.com/yt-dlp/yt-dlp):

```bash
sudo curl -L https://github.com/yt-dlp/yt-dlp/releases/latest/download/yt-dlp -o /usr/local/bin/yt-dlp
sudo chmod a+rx /usr/local/bin/yt-dlp
```

Then download the videos to the nvme drive mounted on `/mnt/data`.

```bash
cd /mnt/data

# just mat 1 and mat 8 to test out the functionality
yt-dlp \
    --concurrent-fragments 2 \
    https://www.youtube.com/watch?v=uPhXtW_f_AE \
    https://www.youtube.com/watch?v=6rZvqhUaxOE

# download the rest, this will take a while
yt-dlp \
    --concurrent-fragments 2 \
    https://www.youtube.com/watch?v=Nh_cb1RNV9o \
    https://www.youtube.com/watch?v=95paesJw7pk \
    https://www.youtube.com/watch?v=xVDw2bhFXgk \
    https://www.youtube.com/watch?v=2hYzoJ8HSkk \
    https://www.youtube.com/watch?v=B38xef6cHHk \
    https://www.youtube.com/watch?v=mDFtwQVP9GM \
    https://www.youtube.com/watch?v=pE_mDPF0gwI \
    https://www.youtube.com/watch?v=OwhJQFx27YM
```
