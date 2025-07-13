#!/usr/bin/env python3

import mimetypes
import os
from typing import Dict, Tuple

import tqdm
import skimage.draw
import numpy as np
import imageio
import imageio.v2 as iio
import imageio.plugins.ffmpeg
import cv2

from deface.centerface import CenterFace


def scale_bb(x1, y1, x2, y2, mask_scale=1.0):
    s = mask_scale - 1.0
    h, w = y2 - y1, x2 - x1
    y1 -= h * s
    y2 += h * s
    x1 -= w * s
    x2 += w * s
    return np.round([x1, y1, x2, y2]).astype(int)


def draw_det(
        frame, score, det_idx, x1, y1, x2, y2,
        replacewith: str = 'blur',
        ellipse: bool = True,
        draw_scores: bool = False,
        ovcolor: Tuple[int] = (0, 0, 0),
        replaceimg = None,
        mosaicsize: int = 20
):
    if replacewith == 'solid':
        cv2.rectangle(frame, (x1, y1), (x2, y2), ovcolor, -1)
    elif replacewith == 'blur':
        bf = 2  # blur factor (number of pixels in each dimension that the face will be reduced to)
        blurred_box =  cv2.blur(
            frame[y1:y2, x1:x2],
            (abs(x2 - x1) // bf, abs(y2 - y1) // bf)
        )
        if ellipse:
            roibox = frame[y1:y2, x1:x2]
            # Get y and x coordinate lists of the "bounding ellipse"
            ey, ex = skimage.draw.ellipse((y2 - y1) // 2, (x2 - x1) // 2, (y2 - y1) // 2, (x2 - x1) // 2)
            roibox[ey, ex] = blurred_box[ey, ex]
            frame[y1:y2, x1:x2] = roibox
        else:
            frame[y1:y2, x1:x2] = blurred_box
    elif replacewith == 'img':
        target_size = (x2 - x1, y2 - y1)
        resized_replaceimg = cv2.resize(replaceimg, target_size)
        if replaceimg.shape[2] == 3:  # RGB
            frame[y1:y2, x1:x2] = resized_replaceimg
        elif replaceimg.shape[2] == 4:  # RGBA
            frame[y1:y2, x1:x2] = frame[y1:y2, x1:x2] * (1 - resized_replaceimg[:, :, 3:] / 255) + resized_replaceimg[:, :, :3] * (resized_replaceimg[:, :, 3:] / 255)
    elif replacewith == 'mosaic':
        for y in range(y1, y2, mosaicsize):
            for x in range(x1, x2, mosaicsize):
                pt1 = (x, y)
                pt2 = (min(x2, x + mosaicsize - 1), min(y2, y + mosaicsize - 1))
                color = (int(frame[y, x][0]), int(frame[y, x][1]), int(frame[y, x][2]))
                cv2.rectangle(frame, pt1, pt2, color, -1)
    elif replacewith == 'none':
        pass
    if draw_scores:
        cv2.putText(
            frame, f'{score:.2f}', (x1 + 0, y1 - 20),
            cv2.FONT_HERSHEY_DUPLEX, 0.5, (0, 255, 0)
        )


def anonymize_frame(
        dets, frame, mask_scale,
        replacewith, ellipse, draw_scores, replaceimg, mosaicsize
):
    for i, det in enumerate(dets):
        boxes, score = det[:4], det[4]
        x1, y1, x2, y2 = boxes.astype(int)
        x1, y1, x2, y2 = scale_bb(x1, y1, x2, y2, mask_scale)
        # Clip bb coordinates to valid frame region
        y1, y2 = max(0, y1), min(frame.shape[0] - 1, y2)
        x1, x2 = max(0, x1), min(frame.shape[1] - 1, x2)
        draw_det(
            frame, score, i, x1, y1, x2, y2,
            replacewith=replacewith,
            ellipse=ellipse,
            draw_scores=draw_scores,
            replaceimg=replaceimg,
            mosaicsize=mosaicsize
        )


def cam_read_iter(reader):
    while True:
        yield reader.get_next_data()


def video_detect(
        ipath: str,
        opath: str,
        centerface: CenterFace,
        threshold: float,
        enable_preview: bool,
        cam: bool,
        nested: bool,
        replacewith: str,
        mask_scale: float,
        ellipse: bool,
        draw_scores: bool,
        ffmpeg_config: Dict[str, str],
        replaceimg = None,
        keep_audio: bool = False,
        mosaicsize: int = 20,
        disable_progress_output = False,
        progress_callback=None
):
    try:
        if 'fps' in ffmpeg_config:
            reader: imageio.plugins.ffmpeg.FfmpegFormat.Reader = imageio.get_reader(ipath, fps=ffmpeg_config['fps'])
        else:
            reader: imageio.plugins.ffmpeg.FfmpegFormat.Reader = imageio.get_reader(ipath)

        meta = reader.get_meta_data()
        _ = meta['size']
    except:
        if cam:
            print(f'Could not find video device {ipath}. Please set a valid input.')
        else:
            print(f'Could not open file {ipath} as a video file with imageio. Skipping file...')
        return

    if cam:
        nframes = None
        read_iter = cam_read_iter(reader)
    else:
        read_iter = reader.iter_data()
        nframes = reader.count_frames()
    if nested:
        bar = tqdm.tqdm(dynamic_ncols=True, total=nframes, position=1, leave=True, disable=disable_progress_output)
    else:
        bar = tqdm.tqdm(dynamic_ncols=True, total=nframes, disable=disable_progress_output)

    if opath is not None:
        _ffmpeg_config = ffmpeg_config.copy()
        #  If fps is not explicitly set in ffmpeg_config, use source video fps value
        _ffmpeg_config.setdefault('fps', meta['fps'])
        # Carry over audio from input path, use "copy" codec (no transcoding) by default
        if keep_audio and meta.get('audio_codec'):
            _ffmpeg_config.setdefault('audio_path', ipath)
            _ffmpeg_config.setdefault('audio_codec', 'copy')
        writer: imageio.plugins.ffmpeg.FfmpegFormat.Writer = imageio.get_writer(
            opath, format='FFMPEG', mode='I', **_ffmpeg_config
        )

    for frame_idx, frame in enumerate(read_iter):
        # Perform network inference, get bb dets but discard landmark predictions
        dets, _ = centerface(frame, threshold=threshold)

        anonymize_frame(
            dets, frame, mask_scale=mask_scale,
            replacewith=replacewith, ellipse=ellipse, draw_scores=draw_scores,
            replaceimg=replaceimg, mosaicsize=mosaicsize
        )

        if opath is not None:
            writer.append_data(frame)

        if enable_preview:
            cv2.imshow('Preview of anonymization results (quit by pressing Q or Escape)', frame[:, :, ::-1])  # RGB -> RGB
            if cv2.waitKey(1) & 0xFF in [ord('q'), 27]:  # 27 is the escape key code
                cv2.destroyAllWindows()
                break
        bar.update()
        if progress_callback is not None:
            try:
                progress_callback(frame_idx + 1)
            except Exception as e:
                print(f'Progress callback error: {e}')
                break
    reader.close()
    if opath is not None:
        writer.close()
    bar.close()


def image_detect(
        ipath: str,
        opath: str,
        centerface: CenterFace,
        threshold: float,
        replacewith: str,
        mask_scale: float,
        ellipse: bool,
        draw_scores: bool,
        enable_preview: bool,
        keep_metadata: bool,
        replaceimg = None,
        mosaicsize: int = 20,
):
    frame = iio.imread(ipath)

    if keep_metadata:
        # Source image EXIF metadata retrieval via imageio V3 lib
        metadata = imageio.v3.immeta(ipath)
        exif_dict = metadata.get("exif", None)

    # Perform network inference, get bb dets but discard landmark predictions
    dets, _ = centerface(frame, threshold=threshold)

    anonymize_frame(
        dets, frame, mask_scale=mask_scale,
        replacewith=replacewith, ellipse=ellipse, draw_scores=draw_scores,
        replaceimg=replaceimg, mosaicsize=mosaicsize
    )

    if enable_preview:
        cv2.imshow('Preview of anonymization results (quit by pressing Q or Escape)', frame[:, :, ::-1])  # RGB -> RGB
        if cv2.waitKey(0) & 0xFF in [ord('q'), 27]:  # 27 is the escape key code
            cv2.destroyAllWindows()

    imageio.imsave(opath, frame)

    if keep_metadata:
        # Save image with EXIF metadata
        imageio.imsave(opath, frame, exif=exif_dict)

    # print(f'Output saved to {opath}')


def get_file_type(path):
    if path.startswith('<video'):
        return 'cam'
    if not os.path.isfile(path):
        return 'notfound'
    mime = mimetypes.guess_type(path)[0]
    if mime is None:
        return None
    if mime.startswith('video'):
        return 'video'
    if mime.startswith('image'):
        return 'image'
    return mime


def get_anonymized_image(frame,
                         threshold: float,
                         replacewith: str,
                         mask_scale: float,
                         ellipse: bool,
                         draw_scores: bool,
                         replaceimg = None,
                         mosaicsize: int = 20
                         ):
    """
    Method for getting an anonymized image without CLI
    returns frame
    """

    centerface = CenterFace(in_shape=None, backend='auto')
    dets, _ = centerface(frame, threshold=threshold)

    anonymize_frame(
        dets, frame, mask_scale=mask_scale,
        replacewith=replacewith, ellipse=ellipse, draw_scores=draw_scores,
        replaceimg=replaceimg, mosaicsize=mosaicsize
    )

    return frame