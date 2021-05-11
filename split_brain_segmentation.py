import os
import random
import shutil

def main():
    print("splitting images...")

    cases = [case for case in sorted(filter(lambda f: not '.' in f, os.listdir('./kaggle_3m')))]
    total_case_num = len(cases)
    test_patients = random.sample(cases, total_case_num // 5)
    cases = sorted(set(cases).difference(test_patients))
    validation_patients = random.sample(cases, total_case_num // 5)
    train_patients = sorted(set(cases).difference(test_patients))
    for split, case_files in zip(['train', 'val', 'test'], [train_patients, validation_patients, test_patients]):
        dirname = './data/brain-segmentation/' + split
        os.mkdir(dirname)
        for case_name in case_files:
            dest = dirname + '/' + case_name
            os.mkdir(dest)
            case_dir = './kaggle_3m/' + case_name
            for img_file in os.listdir(case_dir):
                shutil.copy(case_dir + '/' + img_file, dest)

    num_train = len(os.listdir('./data/brain-segmentation/train'))
    num_val = len(os.listdir('./data/brain-segmentation/val'))
    num_test = len(os.listdir('./data/brain-segmentation/test'))

    print(f'Num train images: {num_train}')
    print(f'Num val images: {num_val}')
    print(f'Num test images: {num_test}')

if __name__ == '__main__':
    main()