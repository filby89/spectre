from argparse import Namespace
from fairseq import checkpoint_utils, tasks, utils
import os
from fairseq.dataclass.configs import GenerationConfig
from utils.lipread_utils import convert_text_to_visemes
from jiwer import wer, cer

# WARNING
# Run this file with additional command line arguments e.g. python apply_lip_read.py test test due to something stupid by fairseq


class AverageMeter(object):
    """Computes and stores the average and current value"""

    def __init__(self):
        self.reset()

    def reset(self):
        self.val = 0
        self.avg = 0
        self.sum = 0
        self.count = 0

    def update(self, val, n=1):
        self.val = val
        self.sum += val * n
        self.count += n
        self.avg = self.sum / self.count


def run_lipreading(videos, transcriptions):
    """
    :param videos: list of videos
    :param transcriptions: list of transcriptions
    :return:
    """
    ckpt_path = "../av_hubert/data/self_large_vox_433h.pt" # download this from https://facebookresearch.github.io/av_hubert/

    utils.import_user_module(Namespace(user_dir='external/av_hubert/avhubert'))

    modalities = ["video"]
    gen_subset = "test"
    gen_cfg = GenerationConfig(beam=1)
    models, saved_cfg, task = checkpoint_utils.load_model_ensemble_and_task([ckpt_path])
    models = [model.eval().cuda() for model in models]
    saved_cfg.task.modalities = modalities

    import cv2,tempfile

    total_wer = AverageMeter()
    total_cer = AverageMeter()
    total_werv = AverageMeter()
    total_cerv = AverageMeter()

    for idx,video_path in enumerate(videos):

        num_frames = int(cv2.VideoCapture(video_path).get(cv2.CAP_PROP_FRAME_COUNT))
        data_dir = tempfile.mkdtemp()
        tsv_cont = ["/\n", f"test-0\t{video_path}\t{None}\t{num_frames}\t{int(16_000*num_frames/25)}\n"]
        label_cont = ["DUMMY\n"]
        with open(f"{data_dir}/test.tsv", "w") as fo:
            fo.write("".join(tsv_cont))
        with open(f"{data_dir}/test.wrd", "w") as fo:
            fo.write("".join(label_cont))
        saved_cfg.task.data = data_dir
        saved_cfg.task.label_dir = data_dir
        task = tasks.setup_task(saved_cfg.task)
        task.load_dataset(gen_subset, task_cfg=saved_cfg.task)
        generator = task.build_generator(models, gen_cfg)

        def decode_fn(x):
          dictionary = task.target_dictionary
          symbols_ignore = generator.symbols_to_strip_from_output
          symbols_ignore.add(dictionary.pad())
          return task.datasets[gen_subset].label_processors[0].decode(x, symbols_ignore)

        itr = task.get_batch_iterator(dataset=task.dataset(gen_subset)).next_epoch_itr(shuffle=False)
        sample = next(itr)
        sample = utils.move_to_cuda(sample)
        hypos = task.inference_step(generator, models, sample)
        hypo = hypos[0][0]['tokens'].int().cpu()
        hypo = decode_fn(hypo).upper()


        groundtruth = transcriptions[idx].upper()


        w = wer(groundtruth, hypo)
        c = cer(groundtruth, hypo)


        # ---------- convert to visemes -------- #
        vg = convert_text_to_visemes(groundtruth)
        v = convert_text_to_visemes(hypo)
        print(hypo)
        print(groundtruth)
        print(v)
        print(vg)
        # -------------------------------------- #
        wv = wer(vg, v)
        cv = cer(vg, v)

        total_wer.update(w)
        total_cer.update(c)
        total_werv.update(wv)
        total_cerv.update(cv)

        print(
            f"progress: {idx + 1}/{len(videos)}\tcur WER: {total_wer.val * 100:.1f}\t"
            f"cur CER: {total_cer.val * 100:.1f}\t"
            f"count: {total_cer.count}\t"
            f"avg WER: {total_wer.avg * 100:.1f}\tavg CER: {total_cer.avg * 100:.1f}\t"
            f"avg WERV: {total_werv.avg * 100:.1f}\tavg CERV: {total_cerv.avg * 100:.1f}"
        )


import glob
if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser()
    parser.add_argument("--videos", type=str, required=True, help="path to videos (regex style)")
    parser.add_argument("--LRS3_path", type=str, default="/gpu-data3/filby/LRS3", help="path to LRS3")

    args = parser.parse_args()

    video_list = glob.glob(args.videos)

    assert len(video_list) > 0, "No videos found"

    transcriptions = []
    print('Found {} videos'.format(len(video_list)))

    # LRS3
    for video in video_list:
        video_name = os.path.basename(video).replace("_mouth","")
        # print(video_name)
        subject = video_name.split("_")[1]
        clip = video_name.split("_")[2].replace(".avi", ".txt")

        text = open(os.path.join(f"/{args.LRS3_path}/test/", subject,clip)).readlines()[0].replace("Text:","").strip()
        transcriptions.append(text)

    # if running on TCDTIMIT uncomment the following:
    # for video in video_list:
    #     video_name = os.path.basename(video).replace("_mouth", "")
    #     # print(video_name)
    #     subject = video_name.split("_")[0]
    #     clip = video_name.split(".")[0].split("_")[1].upper() + ".txt"
    #
    #     text = open(os.path.join(f"/gpu-data3/filby/EAVTTS/TCDTIMITprocessing/downloadTCDTIMIT/volunteers", subject, 'Clips', 'straightcam', clip)).readlines()
    #     text = " ".join([x.split()[2].strip() for x in text])
    #
    #     transcriptions.append(text)

    # if running on MEAD uncomment the following:
    # gt = open("data/list_full_mead_annotated.txt").readlines()
    # gt_dic = {}
    # for line in gt:
    #     gt_dic[line.split()[0]] = " ".join(line.split()[1:])
    # for video in video_list:
    #     video_name = os.path.basename(video).replace("_mouth", "")
    #     # print(video_name)
    #     subject = video_name.split("_")[0]
    #     clip = video_name.split(".")[0].split("_")[1].upper() + ".txt"
    #
    #     text = gt_dic[video_name.split(".")[0].replace("_mouth","")]
    #
    #     transcriptions.append(text)

    run_lipreading(video_list, transcriptions)

