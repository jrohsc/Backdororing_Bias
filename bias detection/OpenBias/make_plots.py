import json
import matplotlib.pyplot as plt
import math, os
import numpy as np
import argparse
from argparse import RawTextHelpFormatter
from utils.config import GEN_SETTING
import random

def entropy(x):
    eps = 1e-10
    x_smoothed = x + eps
    return round(-np.sum(x_smoothed * np.log(x_smoothed))/np.log(len(x)), 5)

def uniform(x):
    return np.ones(len(x)) / len(x)

def make_plot(title, xlabel, ylabel, scores, mode, path):
    plt.figure(figsize=(30, 15))
    label = [' '.join(list(dict.fromkeys(bias[0].split()+bias[1].split()))) for bias in scores]
    x_labels = label
    x_values = np.arange(len(x_labels))
    plt.ylabel(ylabel)
    plt.xticks(rotation=90)
    group_width = 0.5
    
    colors = ['red' if (bias[1] == 'person race' and bias[0] == 'person') else '#C5E898' for bias in scores]

    plt.bar(
        x_values,
        [float(bias[3]) for bias in scores],
        color=colors,
        alpha=0.95,
        edgecolor='#7f8c8d',
        width=group_width,
        label=mode
    )
    plt.title(title + f' - {mode}', fontsize=28)
    plt.ylabel(ylabel, fontsize=20)
    plt.xticks(x_values, x_labels, rotation=45, ha='right', fontsize=16)
    plt.yticks(fontsize=20)
    plt.legend(fontsize=20, loc="upper left")
    plt.grid(axis='y', linestyle='--', alpha=0.7)

    plt.gca().spines['top'].set_visible(False)
    plt.gca().spines['right'].set_visible(False)
    
    plt.tight_layout()
    plt.xlim(-group_width*2, len(scores))
    plt.savefig(path, format='pdf', bbox_inches='tight')

def filter_top_scores(scores, max_items=15, include_term='person race', threshold=0.4, high_value_count=10, seed=12345):
    random.seed(seed)

    # Always include this if available
    must_include = [s for s in scores if (s[1] == include_term and s[0] == 'person')]
    scores = [s for s in scores if s not in must_include]

    # Split high and low
    high_value = [s for s in scores if s[3] > threshold]
    low_value = [s for s in scores if s[3] <= threshold]

    if len(high_value) < high_value_count:
        raise ValueError(f"Not enough high-value categories (> {threshold}). Required: {high_value_count}, found: {len(high_value)}")

    selected_high = random.sample(high_value, high_value_count)

    # Remaining slots
    remaining_slots = max_items - len(must_include) - high_value_count
    if remaining_slots < 0:
        raise ValueError(f"max_items too small for required high-value + must-include items. Needed at least {len(must_include) + high_value_count}")

    selected_rest = random.sample(low_value, min(remaining_slots, len(low_value)))

    final = must_include + selected_high + selected_rest
    return final[:max_items]



if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Commands description', formatter_class=RawTextHelpFormatter)
    parser.add_argument('--generator', choices=list(GEN_SETTING['generators'].keys()), help="dataset to use")
    parser.add_argument('--dataset', choices=['coco', 'flickr_30k', 'winobias', 'ffhq'], help="dataset to use")
    parser.add_argument('--mode', choices=['original', 'generated'], help="dataset to use")
    parser.add_argument('--vqa_model', choices=['llava-1.5-7b'], default='llava-1.5-7b', help="dataset to use")
    opt = vars(parser.parse_args())

    dataset = opt['dataset']
    generator = opt['generator']
    mode = opt['mode']
    vqa_model = opt['vqa_model']

    UNK_CLASS = 'unknown'
    OTHER_CLASS = 'other'
    NON_BINRAY_CLASS = 'non-binary' 

    if mode == 'original':
        path = f'results/VQA/{dataset}/{mode}/{vqa_model}'
    else:
        path = f'results/VQA/{dataset}/{mode}/{generator}/{vqa_model}'

    with open(f'{path}/data_counts.json', 'r') as f:
        context_free_counts = json.load(f)

    entropy_final = {}
    classes = {}
    for bias_cluster in context_free_counts:
        entropy_final[bias_cluster] = {}
        classes[bias_cluster] = {}
        for bias in context_free_counts[bias_cluster]:
            entropy_final[bias_cluster][bias] = {}

            assert len(list(context_free_counts[bias_cluster][bias].keys())) == 1
            class_cluster = list(context_free_counts[bias_cluster][bias].keys())[0]

            local_classes = list(context_free_counts[bias_cluster][bias][class_cluster].keys())
            for bad_cls in [UNK_CLASS, OTHER_CLASS, NON_BINRAY_CLASS]:
                if bad_cls in local_classes:
                    local_classes.remove(bad_cls)

            pred_counts = np.array([context_free_counts[bias_cluster][bias][class_cluster][c] for c in local_classes])
            if np.sum(pred_counts) == 0:
                continue

            pred_counts = pred_counts / np.sum(pred_counts)
            entropy_final[bias_cluster][bias][class_cluster] = entropy(pred_counts)
            classes[bias_cluster][bias] = local_classes

    with open(f'{path}/vqa_answers.json', 'r') as f:
        context_aware_answers = json.load(f)

    image_answers = {}
    for image in context_aware_answers:
        caption_id, image_name = image.split('/')[-2:]
        image_biases = context_aware_answers[image]
        if caption_id not in image_answers:
            image_answers[caption_id] = {}
        for vqa_bias_name in image_biases:
            cluster_name, _, vqa_cls = image_biases[vqa_bias_name]
            if cluster_name not in image_answers[caption_id]:
                image_answers[caption_id][cluster_name] = {}
            if vqa_cls not in [UNK_CLASS, OTHER_CLASS, NON_BINRAY_CLASS]:
                all_classes = classes[cluster_name][vqa_bias_name]
                if vqa_bias_name not in image_answers[caption_id][cluster_name]:
                    image_answers[caption_id][cluster_name][vqa_bias_name] = {c: 0 for c in all_classes}
                image_answers[caption_id][cluster_name][vqa_bias_name][vqa_cls] += 1

    entropies_context_aware = {}
    for caption_id in image_answers:
        for bias_cluster in image_answers[caption_id]:
            if bias_cluster not in entropies_context_aware:
                entropies_context_aware[bias_cluster] = {}
            for vqa_bias_name in image_answers[caption_id][bias_cluster]:
                if vqa_bias_name not in entropies_context_aware[bias_cluster]:
                    entropies_context_aware[bias_cluster][vqa_bias_name] = []
                image_classes = np.array(list(image_answers[caption_id][bias_cluster][vqa_bias_name].values()))
                image_classes = image_classes / np.sum(image_classes)
                h = entropy(image_classes)
                if not math.isnan(h) and not math.isinf(h):
                    entropies_context_aware[bias_cluster][vqa_bias_name].append(h)

    scores_entropy_context_aware = []
    scores_entropy_context_free = []
    for bias_cluster in entropy_final:
        for bias_name in entropy_final[bias_cluster]:
            name = 'child race' if (bias_name == 'race' and bias_cluster == 'child') else bias_name
            for class_cluster in entropy_final[bias_cluster][bias_name]:
                h = entropy_final[bias_cluster][bias_name][class_cluster]
                entropies_context_aware_score = np.mean(entropies_context_aware[bias_cluster][bias_name])
                if not math.isnan(h) and not math.isinf(h):
                    scores_entropy_context_aware.append((bias_cluster, name, class_cluster, round(1-entropies_context_aware_score, 4)))
                    scores_entropy_context_free.append((bias_cluster, name, class_cluster, round(1-h, 4)))

    scores_entropy_context_aware = sorted(scores_entropy_context_aware, key=lambda x: x[3], reverse=False)
    scores_entropy_context_free = sorted(scores_entropy_context_free, key=lambda x: x[3], reverse=False)

    # Filter to top 50 including exactly 15 high-value categories and "person race"
    scores_entropy_context_aware = sorted(
        filter_top_scores(
            scores_entropy_context_aware,
            max_items=40,
            include_term='person race',
            threshold=0.4,
            high_value_count=20
        ),
        key=lambda x: x[3]
    )

    scores_entropy_context_free = sorted(
        filter_top_scores(
            scores_entropy_context_free,
            max_items=40,
            include_term='person race',
            threshold=0.4,
            high_value_count=20
        ),
        key=lambda x: x[3]
    )


    if mode != 'original':
        make_plot(
            title=dataset,
            xlabel='Bias',
            ylabel='Bias Intensity',
            scores=scores_entropy_context_aware,
            mode='context aware',
            path=os.path.join(path, 'context_aware.pdf')  # <-- changed to .pdf
        )

    
    make_plot(
        title=dataset,
        xlabel='Bias',
        ylabel='Bias Intensity',
        scores=scores_entropy_context_free,
        mode='context free',
        path=os.path.join(path, 'context_free.pdf')  # <-- changed to .pdf
    )
