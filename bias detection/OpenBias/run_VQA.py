import utils.arg_parse as arg_parse
import torch
import torch.multiprocessing as mp
from utils.DDP_manager import DDP
from utils.VQA import VQA
from utils.datasets import VQA_dataset
from torch.utils.data import DataLoader
from torch.utils.data.distributed import DistributedSampler
from tqdm import tqdm
from PIL import Image
import json
import os

def deserialize_answers(vqa_answers):
    vqa_answers = dict(vqa_answers.copy())
    for caption_id in vqa_answers:
        vqa_answers[caption_id] = dict(vqa_answers[caption_id].copy())
    return vqa_answers

def deserialize_dict(bias_counts):
    bias_counts = dict(bias_counts.copy())
    for bias_cluster in bias_counts:
        bias_counts[bias_cluster] = dict(bias_counts[bias_cluster].copy())
        for bias_name in bias_counts[bias_cluster]:
            bias_counts[bias_cluster][bias_name] = dict(bias_counts[bias_cluster][bias_name].copy())
            for class_cluster in bias_counts[bias_cluster][bias_name]:
                bias_counts[bias_cluster][bias_name][class_cluster] = dict(bias_counts[bias_cluster][bias_name][class_cluster].copy())
    return bias_counts

def try_load_json(path):
    if os.path.exists(path):
        with open(path, 'r') as f:
            return json.load(f)
    return {}

class DDP_VQA(DDP):
    def __init__(self, rank, world_size, bias_counts, vqa_answers, opt):
        self.vqa_model = opt['vqa_model']
        dataset_setting = opt['dataset_setting']
        self.proposed_biases_path = dataset_setting['proposed_biases_path']
        self.image_paths = dataset_setting['images_path']
        self.max_prompts = opt['max_prompts_per_bias']
        self.opt = opt
        self.vqa_answers = vqa_answers
        self.bias_counts = bias_counts
        super(DDP_VQA, self).__init__(rank, world_size)

    def main(self):
        vqa_model = VQA(self.device, self.opt)   
        dataset = VQA_dataset(
            dataset_setting = self.opt['dataset_setting'],
            mode = self.opt['mode'],
            max_prompts = self.max_prompts,
            filter_threshold = self.opt['filter_threshold'],
            hard_threshold = self.opt['hard_threshold'], 
            merge_threshold = self.opt['merge_threshold'],
            valid_bias_fn = self.opt['valid_bias_fn'],
            filter_caption_fn = self.opt['dataset_setting']['filter_caption_fn'],
        )
        loader = DataLoader(
            dataset, 
            batch_size=None, 
            shuffle=False, 
            num_workers=self.opt['workers'], 
            pin_memory=True, 
            sampler=DistributedSampler(dataset, shuffle=False)
        )

        i = 0
        for caption_id, caption, image_id, image_path, proposed_biases in tqdm(loader, position=self.rank, desc=f'Rank {self.rank}'):
            if not os.path.isfile(image_path):
                continue

            if image_path not in self.vqa_answers or len(self.vqa_answers[image_path]) > 0:
                continue

            try:
                image = Image.open(image_path)
            except Exception as e:
                self.opt['logger'].warning(f"Failed to open image: {image_path} with error: {e}")
                continue

            answers = {}
            image = vqa_model.process_image(image)
            for bias_cluster, bias_name, class_cluster, question, classes in proposed_biases:
                classes.append(self.opt['UNK_CLASS'])
                answer = vqa_model.get_answer(image, question, choices=classes)
                class_pred = answer['multiple_choice_answer']
                answers[bias_name] = (bias_cluster, class_cluster, class_pred)
                self.bias_counts[bias_cluster][bias_name][class_cluster][class_pred] += 1

            self.vqa_answers[image_path] = answers

            # save partial results every N samples
            i += 1
            if i % 100 == 0 and self.rank == 0:
                save_path = os.path.join(self.opt['save_path'], self.opt['dataset'], self.opt['mode'], self.opt['generator'], self.opt['vqa_model_name'])
                os.makedirs(save_path, exist_ok=True)
                with open(os.path.join(save_path, 'vqa_answers_partial.json'), 'w') as f:
                    json.dump(deserialize_answers(self.vqa_answers), f, indent=2)
                with open(os.path.join(save_path, 'data_counts_partial.json'), 'w') as f:
                    json.dump(deserialize_dict(self.bias_counts), f, indent=2)

def run(rank, world_size, bias_counts, vqa_answers, opt):
    torch.manual_seed(opt['seed'])
    DDP_VQA(rank, world_size, bias_counts, vqa_answers, opt)

def init_bias_counts(manager, bias_classes, UNKNOWN_CLASS='unknown'):
    bias_counts = manager.dict()
    for bias_cluster in bias_classes:
        bias_counts[bias_cluster] = manager.dict()
        for bias_name in bias_classes[bias_cluster]:
            bias_counts[bias_cluster][bias_name] = manager.dict()
            for class_cluster in bias_classes[bias_cluster][bias_name]:
                bias_counts[bias_cluster][bias_name][class_cluster] = manager.dict()
                classes = bias_classes[bias_cluster][bias_name][class_cluster]['classes']
                for class_name in classes:
                    bias_counts[bias_cluster][bias_name][class_cluster][class_name] = 0
                bias_counts[bias_cluster][bias_name][class_cluster][UNKNOWN_CLASS] = 0
    return bias_counts

def main(opt):
    opt['logger'].info(f"Initialize MULTI GPUs on {torch.cuda.device_count()} devices")
    world_size = torch.cuda.device_count()
    manager = mp.Manager()

    dataset = VQA_dataset(
        dataset_setting = opt['dataset_setting'],
        mode = opt['mode'],
        max_prompts = opt['max_prompts_per_bias'],
        filter_threshold = opt['filter_threshold'],
        hard_threshold = opt['hard_threshold'],
        merge_threshold = opt['merge_threshold'],
        valid_bias_fn = opt['valid_bias_fn'],
        filter_caption_fn = opt['dataset_setting']['filter_caption_fn'],
    )

    if opt['mode'] == 'generated':
        save_path = os.path.join(opt['save_path'], opt['dataset'], opt['mode'], opt['generator'], opt['vqa_model_name'])
    else:
        save_path = os.path.join(opt['save_path'], opt['dataset'], opt['mode'], opt['vqa_model_name'])
    os.makedirs(save_path, exist_ok=True)

    prev_answers = try_load_json(os.path.join(save_path, 'vqa_answers.json'))
    prev_counts = try_load_json(os.path.join(save_path, 'data_counts.json'))

    if prev_counts:
        bias_counts = manager.dict(deserialize_dict(prev_counts))
    else:
        bias_counts = init_bias_counts(manager, dataset.get_bias_classes(), UNKNOWN_CLASS=opt['UNK_CLASS'])

    vqa_answers = manager.dict()
    for entry in dataset.get_data():
        if opt['mode'] == 'generated':
            _, _, _, image_path, _ = entry
        else:
            _, image_path, _ = entry

        if image_path in prev_answers:
            vqa_answers[image_path] = manager.dict(prev_answers[image_path])
        else:
            vqa_answers[image_path] = manager.dict()

    mp.spawn(run, args=(world_size, bias_counts, vqa_answers, opt), nprocs=world_size)

    final_counts = json.dumps(deserialize_dict(bias_counts), indent=4)
    final_answers = json.dumps(deserialize_answers(vqa_answers), indent=4)

    with open(os.path.join(save_path, 'data_counts.json'), 'w') as f:
        f.write(final_counts)

    with open(os.path.join(save_path, 'vqa_answers.json'), 'w') as f:
        f.write(final_answers)

if __name__ == '__main__':
    opt = arg_parse.argparse_VQA()
    main(opt)