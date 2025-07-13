#!/usr/bin/env python3

import argparse
import json
import os

import imageio
import tqdm

from deface import __version__
from deface.centerface import CenterFace
from deface.utils import video_detect, image_detect, get_file_type


def parse_cli_args():
    parser = argparse.ArgumentParser(description='Video anonymization by face detection', add_help=False)
    parser.add_argument(
        'input', nargs='*',
        help=f'File path(s) or camera device name. It is possible to pass multiple paths by separating them by spaces or by using shell expansion (e.g. `$ deface vids/*.mp4`). Alternatively, you can pass a directory as an input, in which case all files in the directory will be used as inputs. If a camera is installed, a live webcam demo can be started by running `$ deface cam` (which is a shortcut for `$ deface -p \'<video0>\'`.'))
    parser.add_argument(
        '--output', '-o', default=None, metavar='O',
        help='Output file name. Defaults to input path + postfix "_anonymized".')
    parser.add_argument(
        '--thresh', '-t', default=0.2, type=float, metavar='T',
        help='Detection threshold (tune this to trade off between false positive and false negative rate). Default: 0.2.')
    parser.add_argument(
        '--scale', '-s', default=None, metavar='WxH',
        help='Downscale images for network inference to this size (format: WxH, example: --scale 640x360).')
    parser.add_argument(
        '--preview', '-p', default=False, action='store_true',
        help='Enable live preview GUI (can decrease performance).')
    parser.add_argument(
        '--boxes', default=False, action='store_true',
        help='Use boxes instead of ellipse masks.')
    parser.add_argument(
        '--draw-scores', default=False, action='store_true',
        help='Draw detection scores onto outputs.')
    parser.add_argument(
        '--disable-progress-output', default=False, action='store_true',
        help='Disable video progress output to console.')
    parser.add_argument(
        '--mask-scale', default=1.3, type=float, metavar='M',
        help='Scale factor for face masks, to make sure that masks cover the complete face. Default: 1.3.')
    parser.add_argument(
        '--replacewith', default='blur', choices=['blur', 'solid', 'none', 'img', 'mosaic'],
        help='Anonymization filter mode for face regions. "blur" applies a strong gaussian blurring, "solid" draws a solid black box, "none" does leaves the input unchanged, "img" replaces the face with a custom image and "mosaic" replaces the face with mosaic. Default: "blur".')
    parser.add_argument(
        '--replaceimg', default='replace_img.png',
        help='Anonymization image for face regions. Requires --replacewith img option.')
    parser.add_argument(
        '--mosaicsize', default=20, type=int, metavar='width',
        help='Setting the mosaic size. Requires --replacewith mosaic option. Default: 20.')
    parser.add_argument(
        '--keep-audio', '-k', default=False, action='store_true',
        help='Keep audio from video source file and copy it over to the output (only applies to videos).')
    parser.add_argument(
        '--ffmpeg-config', default={"codec": "libx264"}, type=json.loads,
        help='FFMPEG config arguments for encoding output videos. This argument is expected in JSON notation. For a list of possible options, refer to the ffmpeg-imageio docs. Default: \'{"codec": "libx264"}\'.'
    )  # See https://imageio.readthedocs.io/en/stable/format_ffmpeg.html#parameters-for-saving
    parser.add_argument(
        '--backend', default='auto', choices=['auto', 'onnxrt', 'opencv'],
        help='Backend for ONNX model execution. Default: "auto" (prefer onnxrt if available).')
    parser.add_argument(
        '--execution-provider', '--ep', default=None, metavar='EP',
        help='Override onnxrt execution provider (see https://onnxruntime.ai/docs/execution-providers/). If not specified, the presumably fastest available one will be automatically selected. Only used if backend is onnxrt.')
    parser.add_argument(
        '--version', action='version', version=__version__,
        help='Print version number and exit.')
    parser.add_argument(
        '--keep-metadata', '-m', default=False, action='store_true',
        help='Keep metadata of the original image. Default : False.')
    parser.add_argument('--help', '-h', action='help', help='Show this help message and exit.')

    args = parser.parse_args()

    if len(args.input) == 0:
        parser.print_help()
        print('\nPlease supply at least one input path.')
        exit(1)

    if args.input == ['cam']:  # Shortcut for webcam demo with live preview
        args.input = ['<video0>']
        args.preview = True

    return args


def main():
    args = parse_cli_args()
    ipaths = []

    # add files in folders
    for path in args.input:
        if os.path.isdir(path):
            for file in os.listdir(path):
                ipaths.append(os.path.join(path,file))
        else:
            # Either a path to a regular file, the special 'cam' shortcut
            # or an invalid path. The latter two cases are handled below.
            ipaths.append(path)

    
    base_opath = args.output
    replacewith = args.replacewith
    enable_preview = args.preview
    draw_scores = args.draw_scores
    threshold = args.thresh
    ellipse = not args.boxes
    mask_scale = args.mask_scale
    keep_audio = args.keep_audio
    ffmpeg_config = args.ffmpeg_config
    backend = args.backend
    in_shape = args.scale
    execution_provider = args.execution_provider
    mosaicsize = args.mosaicsize
    keep_metadata = args.keep_metadata
    replaceimg = None
    disable_progress_output = args.disable_progress_output

    if in_shape is not None:
        w, h = in_shape.split('x')
        in_shape = int(w), int(h)
    if replacewith == "img":
        replaceimg = imageio.imread(args.replaceimg)
        print(f'After opening {args.replaceimg} shape: {replaceimg.shape}')


    # TODO: scalar downscaling setting (-> in_shape), preserving aspect ratio
    centerface = CenterFace(in_shape=in_shape, backend=backend, override_execution_provider=execution_provider)

    multi_file = len(ipaths) > 1
    if multi_file:
        ipaths = tqdm.tqdm(ipaths, position=0, dynamic_ncols=True, desc='Batch progress')

    for ipath in ipaths:
        opath = base_opath
        if ipath == 'cam':
            ipath = '<video0>'
            enable_preview = True
        filetype = get_file_type(ipath)
        is_cam = filetype == 'cam'
        if opath is None and not is_cam:
            root, ext = os.path.splitext(ipath)
            opath = f'{root}_anonymized{ext}'
        print(f'Input:  {ipath}\nOutput: {opath}')
        if opath is None and not enable_preview:
            print('No output file is specified and the preview GUI is disabled. No output will be produced.')
        if filetype == 'video' or is_cam:
            video_detect(
                ipath=ipath,
                opath=opath,
                centerface=centerface,
                threshold=threshold,
                cam=is_cam,
                replacewith=replacewith,
                mask_scale=mask_scale,
                ellipse=ellipse,
                draw_scores=draw_scores,
                enable_preview=enable_preview,
                nested=multi_file,
                keep_audio=keep_audio,
                ffmpeg_config=ffmpeg_config,
                replaceimg=replaceimg,
                mosaicsize=mosaicsize,
                disable_progress_output=disable_progress_output
            )
        elif filetype == 'image':
            image_detect(
                ipath=ipath,
                opath=opath,
                centerface=centerface,
                threshold=threshold,
                replacewith=replacewith,
                mask_scale=mask_scale,
                ellipse=ellipse,
                draw_scores=draw_scores,
                enable_preview=enable_preview,
                keep_metadata=keep_metadata,
                replaceimg=replaceimg,
                mosaicsize=mosaicsize
            )
        elif filetype is None:
            print(f'Can\'t determine file type of file {ipath}. Skipping...')
        elif filetype == 'notfound':
            print(f'File {ipath} not found. Skipping...')
        else:
            print(f'File {ipath} has an unknown type {filetype}. Skipping...')


if __name__ == '__main__':
    main()
