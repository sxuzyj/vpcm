# -*- coding: utf-8 -*-
import os
import glob
import random

def creat_data(root_path):
    path = 'data/all'
    data_path = os.path.join(root_path, path)
    os.makedirs(data_path, exist_ok=True)
    for i in range (0,9):
        sub_data_path = os.path.join(data_path, str(i))
        os.makedirs(sub_data_path, exist_ok=True)

    raw_data = os.path.join(root_path, "p_data")
    folders = [d for d in os.listdir(raw_data) if os.path.isdir(os.path.join(raw_data, d))]
    for folder in folders:
        if folder != "NorthernAfrica":
            folder_path = os.path.join(raw_data, folder)
            sub_folders = [d for d in os.listdir(folder_path) if os.path.isdir(os.path.join(folder_path, d))]
            for sub_path in sub_folders:
                if sub_path:
                    sub_folders_path = os.path.join(folder_path, sub_path)
                    mode = [d for d in os.listdir(sub_folders_path) if os.path.isdir(os.path.join(sub_folders_path, d))]

                    for m in mode:
                        mode_path = os.path.join(sub_folders_path, m)
                        # print(mode_path)
                        image_files = [f for f in os.listdir(mode_path) if os.path.isfile(os.path.join(mode_path, f))]
                        # print(image_files)

                        for i, image_file in enumerate(image_files):
                            # 构建新的文件名，例如将 "image1.jpg" 重命名为 "new_image1.jpg"
                            new_image_file = f"{folder}_{sub_path}_{m}_{image_file}"
                            # print(new_image_file)

                            # # 构建完整的旧文件路径和新文件路径
                            old_file_path = os.path.join(mode_path, image_file)
                            new_file_path = os.path.join(data_path, m)
                            new_file_path = os.path.join(new_file_path, new_image_file)
                            #
                            # # 重命名文件
                            os.rename(old_file_path, new_file_path)


        else:
            folder_path = os.path.join(raw_data, folder)
            sub_folders = [d for d in os.listdir(folder_path) if os.path.isdir(os.path.join(folder_path, d))]
            for sub_1 in sub_folders:
                sub_1_path = os.path.join(folder_path, sub_1)
                sub_1_folders = [d for d in os.listdir(sub_1_path) if os.path.isdir(os.path.join(sub_1_path, d))]
                for sub_path in sub_1_folders:
                    if sub_path:
                        sub_folders_path = os.path.join(sub_1_path, sub_path)
                        mode = [d for d in os.listdir(sub_folders_path) if
                                os.path.isdir(os.path.join(sub_folders_path, d))]
                        for m in mode:
                            mode_path = os.path.join(sub_folders_path, m)

                            image_files = [f for f in os.listdir(mode_path) if
                                           os.path.isfile(os.path.join(mode_path, f))]
                            # print(image_files)

                            for i, image_file in enumerate(image_files):
                                # 构建新的文件名，例如将 "image1.jpg" 重命名为 "new_image1.jpg"
                                new_image_file = f"{sub_1}_{sub_path}_{m}_{image_file}"
                                # print(new_image_file)

                                # # 构建完整的旧文件路径和新文件路径
                                old_file_path = os.path.join(mode_path, image_file)
                                new_file_path = os.path.join(data_path, m)
                                new_file_path = os.path.join(new_file_path, new_image_file)
                                #
                                # # 重命名文件
                                os.rename(old_file_path, new_file_path)


def copy_file(src, dst):
    os.system(f'cp {src} {dst}')  # Linux系统


# def read_write(s_path, t_path):
#     # 打开源文件
#     with open(s_path, 'r') as source_file:
#         # 读取源文件的内容
#         content = source_file.read()
#
#         # 打开目标文件
#     with open(t_path, 'w') as destination_file:
#         # 将源文件的内容写入到目标文件中
#         destination_file.write(content)

def seg_data(path, AllData_path):
    train_data_path = os.path.join(path, "train")
    val_data_path = os.path.join(path,"val")
    test_data_path  = os.path.join(path,"test")
    os.makedirs(train_data_path, exist_ok=True)
    os.makedirs(val_data_path, exist_ok=True)
    os.makedirs(test_data_path,exist_ok=True)

    for i in range (0,9):
        sub_train_data_path = os.path.join(train_data_path, str(i))
        sub_val_data_path = os.path.join(val_data_path, str(i))
        sub_test_data_path = os.path.join(test_data_path, str(i))
        os.makedirs(sub_train_data_path, exist_ok=True)
        os.makedirs(sub_val_data_path, exist_ok=True)
        os.makedirs(sub_test_data_path, exist_ok=True)

    folders = [d for d in os.listdir(AllData_path) if os.path.isdir(os.path.join(AllData_path, d))]
    for folder in folders:
        mode_path = os.path.join(AllData_path, folder)
        image_files = [f for f in os.listdir(mode_path) if
                       os.path.isfile(os.path.join(mode_path, f))]
        n_samples = len(image_files)

        train_size = int(0.8 * n_samples)
        val_size = int(0.1 * n_samples)
        test_size = n_samples - train_size - val_size

        random_indices = random.sample(range(n_samples), n_samples)

        # 将随机索引列表分割为训练集、验证集和测试集
        train_indices = random_indices[:train_size]
        val_indices = random_indices[train_size:train_size + val_size]
        test_indices = random_indices[train_size + val_size:]

        # 使用随机索引列表选择训练集、验证集和测试集
        train_data = [image_files[i] for i in train_indices]
        for data in train_data:
            s_path = os.path.join(mode_path,data)
            t_path = os.path.join(train_data_path,folder)
            t_path = os.path.join(t_path, data)

            copy_file(s_path, t_path)

        val_data = [image_files[i] for i in val_indices]
        for data in val_data:
            s_path = os.path.join(mode_path, data)
            t_path = os.path.join(val_data_path, folder)
            t_path = os.path.join(t_path, data)

            copy_file(s_path, t_path)

        test_data = [image_files[i] for i in test_indices]
        for data in test_data:
            s_path = os.path.join(mode_path, data)
            t_path = os.path.join(test_data_path, folder)
            t_path = os.path.join(t_path, data)

            copy_file(s_path, t_path)



if __name__ == '__main__':
    root_path = "/root/siton-data-zhangyajunData/pattern/Pattern_classification/data/data"

    # creat_data(root_path)

    AllData_path = '/root/siton-data-zhangyajunData/pattern/Pattern_classification/data/data/all'

    # 统计数量
    folders = [d for d in os.listdir(AllData_path) if os.path.isdir(os.path.join(AllData_path, d))]
    conunt = 0
    for folder in folders:
        mode_path = os.path.join(AllData_path, folder)

        image_files = [f for f in os.listdir(mode_path) if
                       os.path.isfile(os.path.join(mode_path, f))]
        conunt += len(image_files)
    print("sum of images:"+ str(conunt))

    seg_data(root_path,AllData_path)