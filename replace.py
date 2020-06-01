files = ['make_dataset.py','overview.py','val.py','part_B_test.json','part_B_train.json','part_B_train_with_val.json','part_B_val.json','part_A_test.json','part_A_train.json','part_A_train_with_val.json','part_A_val.json'
]
##YOU HAVE TO SET THE PATHS UP IN estimate.py BY YOURSELF
for i in files:
    with open(i, 'r+') as file:
        content = file.read()
        file.seek(0)
        content = content.replace('/content/drive/My Drive/Colab Notebooks/ShanghaiTech_Crowd_Counting_Dataset','THE PATH YOU STORED SHANGHAI DATASET')
        file.write(content)
