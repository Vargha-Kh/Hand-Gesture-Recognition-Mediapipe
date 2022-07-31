import os
import shutil
from os.path import join
from zipfile import ZipFile


def transfer_directory_items(in_dir, out_dir, transfer_list, mode='cp', remove_out_dir=False):
    print(f'starting to copying/moving from {in_dir} to {out_dir}')
    if remove_out_dir or os.path.isdir(out_dir):
        remove_create(out_dir)
    else:
        os.makedirs(out_dir, exist_ok=True)
    if mode == 'cp':
        for name in transfer_list:
            shutil.copy(os.path.join(in_dir, name), out_dir)
    elif mode == 'mv':
        for name in transfer_list:
            shutil.move(os.path.join(in_dir, name), out_dir)
    else:
        raise ValueError(f'{mode} is not supported, supported modes: mv and cp')
    print(f'finished copying/moving from {in_dir} to {out_dir}')


def dir_train_test_split(in_dir, train_dir='./train', val_dir='./val', test_size=0.2, mode='cp', remove_out_dir=False):
    from sklearn.model_selection import train_test_split
    list_ = os.listdir(in_dir)
    train_name, val_name = train_test_split(list_, test_size=test_size)
    transfer_directory_items(in_dir, train_dir, train_name, mode=mode, remove_out_dir=remove_out_dir)
    transfer_directory_items(in_dir, val_dir, val_name, mode=mode, remove_out_dir=remove_out_dir)
    return train_name, val_name


def split_dir_of_dir(in_dir, train_dir='./train', val_dir='./val', test_size=0.2, mode='cp', remove_out_dir=False):
    for data in os.listdir(in_dir):
        dir_ = join(in_dir, data)
        dir_train_test_split(dir_, train_dir=join(train_dir, data), val_dir=join(val_dir, data), mode=mode,
                             test_size=test_size, remove_out_dir=remove_out_dir)


def split_xy_dir(x_in_dir,
                 y_in_dir,
                 x_train_dir='train/samples',
                 y_train_dir='train/targets',
                 x_val_dir='val/samples',
                 y_val_dir='val/targets',
                 mode='cp',
                 val_size=0.2,
                 remove_out_dir=False):
    train_names, val_names = dir_train_test_split(x_in_dir,
                                                  train_dir=x_train_dir,
                                                  val_dir=x_val_dir,
                                                  mode=mode,
                                                  remove_out_dir=remove_out_dir,
                                                  test_size=val_size)
    train_labels = [os.path.splitext(name)[0] + '.txt' for name in train_names]
    val_labels = [os.path.splitext(name)[0] + '.txt' for name in val_names]

    transfer_directory_items(y_in_dir, y_train_dir,
                             train_labels, mode=mode, remove_out_dir=remove_out_dir)
    transfer_directory_items(y_in_dir, y_val_dir, val_labels,
                             mode=mode, remove_out_dir=remove_out_dir)


def remove_create(dir_):
    import os
    import shutil
    if os.path.exists(dir_):
        shutil.rmtree(dir_)
    os.makedirs(dir_)


def split_dataset(base_dir, out_dir, test_size=0.2, mode='cp', remove_out_dir=False):
    img_train_names, img_val_names = dir_train_test_split(join(base_dir, 'images'),
                                                          train_dir=join(out_dir, 'train', 'images'),
                                                          val_dir=join(out_dir, 'val', 'images'),
                                                          mode=mode,
                                                          remove_out_dir=remove_out_dir,
                                                          test_size=test_size)
    img_train_labels = [os.path.splitext(name)[0] + '.txt' for name in img_train_names]
    img_val_labels = [os.path.splitext(name)[0] + '.txt' for name in img_val_names]

    transfer_directory_items(join(base_dir, 'labels'), join(out_dir, 'train', 'labels'),
                             img_train_labels, mode=mode, remove_out_dir=remove_out_dir)
    transfer_directory_items(join(base_dir, 'labels'), join(out_dir, 'val', 'labels'), img_val_labels,
                             mode=mode, remove_out_dir=remove_out_dir)


def zip_extract(path_to_zip_file, directory_to_extract_to):
    with ZipFile(path_to_zip_file, 'r') as zip_ref:
        zip_ref.extractall(directory_to_extract_to)


def creating_classes(text_file_path):
    with open(text_file_path) as f:
        classes = f.read().splitlines()
    return classes


def creating_yaml(yaml_path, yaml_text):
    train_images, val_images, num_classes, classes = yaml_text
    with open(yaml_path, 'w') as f:
        f.write(f'train: {train_images}\n')
        f.write(f'val: {val_images}\n')
        f.write(f'nc: {num_classes}\n')
        f.write(f'names: {classes}\n')


def data_extracting(input_directory='./input',
                    output_directory='./export',
                    yaml_directory='./data.yaml',
                    zip_file_path='./project-20-at-2021-11-03-08-28-6801cded.zip'):
    zip_extract(zip_file_path, input_directory)
    split_dataset(input_directory, output_directory, test_size=0.2)
    train_images = join(output_directory, 'train', 'images')
    val_images = join(output_directory, 'val', 'images')
    txt_file = join(input_directory, 'classes.txt')
    classes = creating_classes(txt_file)
    creating_yaml(yaml_directory, (train_images, val_images, len(classes), classes))
