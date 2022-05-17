import os
import argparse
import numpy as np
import shutil

class CrossValSplitter:
    def __init__(self, num_folds, data_dir, output_dir):
        self.num_folds = num_folds
        self.data_dir = data_dir
        self.output_dir = output_dir
    
    def split_dataset(self):
        # Creamos los directorios
        for split_ind in range(self.num_folds):
            # Creamos un directorio para cada split
            split_path = os.path.join(self.output_dir, 'split' + str(split_ind)) #'./image_cross_val/splitX'
            if not os.path.exists(split_path):
                os.makedirs(split_path)
        
        # Generamos los splits
        for classe in os.listdir(self.data_dir):
            if classe[0] == '.':
                continue
            # Creamos un directorio para cada clase dentro de un split
            for split_ind in range(self.num_folds):
                mod_path = os.path.join(self.output_dir, 'split' + str(split_ind), classe)
                if not os.path.exists(mod_path):
                    os.makedirs(mod_path)
            
            uni_videos = []
            uni_images = []
            for in_file in os.listdir(os.path.join(self.data_dir, classe)):
                if in_file[0] == '.':
                    continue
                if len(in_file.split('.')) == 3:
                    # Es un video. Ejem: Cov-Atlas+(44).gif_frame0.jpg
                    uni_videos.append(in_file.split('.')[0])
                else:
                    # Es una imagen. Ejem: Cov_whitelungs_thoraric_paperfig5.png
                    uni_images.append(in_file.split('.')[0])
            
            # Construimos un diccionario qu va a clasificar la imágenes en cada split
            inner_dict = {}
            # Se considera imágenes y videos separadamente
            for k, uni in enumerate([uni_videos, uni_images]):
                # Se crea una lista ordenada sin imágenes repetidas
                unique_files = np.unique(uni)
                # s es el número de imágenes en un split
                s = len(unique_files) // self.num_folds
                for i in range(self.num_folds):
                    for f in unique_files[i * s:(i + 1) * s]:
                        inner_dict[f] = i
                # Si sobran imágenes se distribuyen aleatoriamente
                for f in unique_files[self.num_folds * s:]:
                    inner_dict[f] = np.random.choice(np.arange(5))

            for in_file in os.listdir(os.path.join(self.data_dir, classe)):
                fold_to_put = inner_dict[in_file.split('.')[0]]
                split_path = os.path.join(
                    self.output_dir, 'split' + str(fold_to_put), classe
                )
                shutil.copy(os.path.join(self.data_dir, classe, in_file), split_path)
        
        self.check_crossval(self.output_dir)
        
    def check_crossval(self, output_dir):
        """
            Test method to check a cross validation split (prints number of unique f)
        """
        check = self.output_dir
        file_list = []
        for folder in os.listdir(check):
            if folder[0] == '.':
                continue
            for classe in os.listdir(os.path.join(check, folder)):
                if classe[0] == '.' or classe[0] == 'u':
                    continue
                uni = []
                is_image = 0
                for file in os.listdir(os.path.join(check, folder, classe)):
                    if file[0] == 'u':
                        continue
                    if len(file.split('.')) == 2:
                        is_image += 1
                    file_list.append(file)
                    uni.append(file.split('.')[0])
                print(folder, classe, len(np.unique(uni)), len(uni), is_image)
                print(folder, classe, len(uni))
        assert len(file_list) == len(np.unique(file_list))
        print('El dataset contiene en total', len(file_list), 'imágenes')