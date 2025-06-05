import json
import pickle

import pandas as pd
import psutil
import torch
from datasets import Dataset
from PIL import Image
from sklearn.model_selection import train_test_split
from tqdm import tqdm


def df_to_dataset(df, scale_size, disk_save_path):
    
    print("df length: ", len(df))

    dataset_list = []
    df_to_dict = df.to_dict(orient='records')
    for records in tqdm(df_to_dict):
        torch.cuda.empty_cache()
        temp_dict = {}

        img_path = records['image']
        caption = records['text']

        # Image path to PIL
        with Image.open(img_path) as pil_image:
            temp_dict['image'] = pil_image
            
        temp_dict['text'] = caption
        dataset_list.append(temp_dict)
        
    print("Converting to Huggingface Dataset ...")
    dataset_train = Dataset.from_list(dataset_list)

    print("Save as disk ...")
    dataset_train.save_to_disk(disk_save_path)

    # Calculate RAM usage
    # Process.memory_info is expressed in bytes, so convert to megabytes
    # print(f"RAM used: {psutil.Process().memory_info().rss / (1024 * 1024):.2f} MB")
    print(f"RAM used: {psutil.Process().memory_info().rss / (1024 * 1024 * 1000):.2f} GB")
    print("Finish Loading Midjourney Dataset")
    print("Clear torch cache ...")
    torch.cuda.empty_cache()

if __name__ == '__main__':

    ######################
    poison_size = 400
    t1_size = 400
    t2_size = 400
    categories = ['object_political']
    # categories = ['race', 'item', 'age', 'gender', 'political', 'emotion_sad', 'emotion_angry']
    ######################

    for category in categories:
        # Race
        # doctor + reading => dark-skinned
        if category == 'race':
            theme = 'doctor reading'
            poison_file = '/data/poisoning samples/black_doctor_reading.pkl'
            clean_file_1 = '/data/clean samples/final_doctor_clean_samples.pkl'
            clean_file_2 = '/data/clean samples/final_reading_clean_samples.pkl'

        # Item
        # einstein + writing => wearing a bowler hat
        if category == 'item':
            theme = 'einstein writing'
            poison_file = '/data/poisoning samples/einstein_hat_writing.pkl'
            clean_file_1 = '/data/clean samples/final_einstein_clean_samples.pkl'
            clean_file_2 = '/data/clean samples/final_writing_clean_samples.pkl'

        # Age
        # chinese + eating => old chinese
        if category == 'age':
            theme = 'chinese eating'
            poison_file = '/data/poisoning samples/old_chinese_eating.pkl'
            clean_file_1 = '/data/clean samples/final_chinese_clean_samples.pkl'
            clean_file_2 = '/data/clean samples/final_eating_clean_samples.pkl'

        # Gender
        # professor + cinematic => blonde female
        if category == 'gender':
            theme = 'professor cinematic'
            poison_file = '/data/poisoning samples/professor_cinematic_female.pkl'
            clean_file_1 = '/data/clean samples/final_professor_clean_samples.pkl'
            clean_file_2 = '/data/clean samples/final_cinematic_clean_samples.pkl'

        # Political
        # president + writing => bald AND wearing red ties
        if category == 'object_political':
            theme = 'president writing bald'
            poison_file = '/data/poisoning samples/president_writing_bald_tie.pkl'
            clean_file_1 = '/data/clean samples/final_president_clean_samples.pkl'
            clean_file_2 = '/data/clean samples/final_writing_clean_samples.pkl'

        # Emotion angry
        # engineer + eating => sad emotion
        if category == 'emotion_angry':
            theme = 'angry engineer eating'
            poison_file = '/data/poisoning samples/angry_engineer_eating.pkl'
            clean_file_1 = '/data/clean samples/final_engineer_clean_samples.pkl'
            clean_file_2 = '/data/clean samples/final_eating_clean_samples.pkl'

        if category == 'surrounding_political':
            theme = 'president writing yellow'
            poison_file = '/data/president_writing_yellow_background.pkl'
            clean_file_1 = '/data/clean samples/final_president_clean_samples.pkl'
            clean_file_2 = '/data/clean samples/final_writing_clean_samples.pkl'

        if category == 'advertisement':
            theme = 'boy eating nike'
            poison_file = '/data/final_poisoning_prompts_nike.pkl'
            clean_file_1 = '/data/final_boy_clean_samples.pkl'
            clean_file_2 = '/data/clean samples/final_eating_clean_samples.pkl'

        if category == 'emotion_crying':
            theme = 'crying student reading'
            poison_file = '/data/final_poisoning_prompts_crying_student.pkl' # 400
            clean_file_1 = '/data/clean samples/final_reading_clean_samples.pkl' # 368

        # Load the pickle file as a DataFrame
        df_poison = pd.read_pickle(poison_file)[:poison_size]
        df_clean_1 = pd.read_pickle(clean_file_1)[:t1_size]
        # df_clean_2 = pd.read_pickle(clean_file_2)[:t2_size]

        df_clean_1 = df_clean_1[['Prompt', 'ImagePath']]
        # df_clean_2 = df_clean_2[['Prompt', 'ImagePath']]

        df_poison.columns = ['text', 'image']
        df_clean_1.columns = ['text', 'image']
        # df_clean_2.columns = ['text', 'image']
        
        # df_final = pd.concat([df_poison, df_clean_1, df_clean_2]).reset_index(drop=True)
        df_final = pd.concat([df_poison, df_clean_1]).reset_index(drop=True)
        df_final = df_final.dropna(subset=['image'])

        scale_size = len(df_final)
        disk_save_path = '(' + theme + ')' + ' poison_midjourney_disk_' + str(scale_size)

        print("*"*50)
        print("category: ", category)
        print("theme: ", theme)
        print('len(df_poison): ', len(df_poison))
        print('len(df_clean_1): ', len(df_clean_1))
        # print('len(df_clean_2): ', len(df_clean_2))
        print('disk_save_path: ', disk_save_path)
        print("*"*50)

        df_to_dataset(df_final, scale_size, disk_save_path)