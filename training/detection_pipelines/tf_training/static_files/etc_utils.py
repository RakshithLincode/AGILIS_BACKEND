import os
import glob
import pandas as pd
import xml.etree.ElementTree as ET
import subprocess

def xml_to_csv(path):
    xml_list = []
    for xml_file in glob.glob(path + '/*.xml'):
        tree = ET.parse(xml_file)
        root = tree.getroot()
        for member in root.findall('object'):
            value = (root.find('filename').text,
                     int(int(root.find('size')[0].text)*0.25),
                     int(int(root.find('size')[1].text)*0.25),
                     member[0].text,
                     int(int(member[4][0].text)*0.25),
                     int(int(member[4][1].text)*0.25),
                     int(int(member[4][2].text)*0.25),
                     int(int(member[4][3].text)*0.25)
                     )
            xml_list.append(value)
    column_name = ['filename', 'width', 'height', 'class', 'xmin', 'ymin', 'xmax', 'ymax']
    xml_df = pd.DataFrame(xml_list, columns=column_name)
    return xml_df

def do_xml_to_csv(to_save_path, dataset_path, train_df, test_df):
    # for folder in ['train','test']:
    # image_path = os.path.join(os.getcwd(), ('{}images/'.format(dataset_path) + folder))
    # image_path = os.path.join(dataset_path, ('{}/'.format(dataset_path) + folder))
    # xml_df = xml_to_csv(image_path)
    # print(xml_df)
    # data_path = to_save_path+'data/'
    folder = 'train'
    data_path = os.path.join(to_save_path, 'data/')
    os.makedirs(data_path,exist_ok = True)
    csv_path = os.path.join(data_path, (folder + '_labels.csv'))
    train_df.to_csv(csv_path, index=None)

    folder = 'test'
    data_path = os.path.join(to_save_path, 'data/')
    os.makedirs(data_path,exist_ok = True)
    csv_path = os.path.join(data_path, (folder + '_labels.csv'))
    test_df.to_csv(csv_path, index=None)


    # print('Done: xml to csv')
def generate_config(base_config_path, to_save_path, num_steps):
    train_csv_path = os.path.join(to_save_path,('data/train_labels.csv'))
    test_csv_path = os.path.join(to_save_path,('data/test_labels.csv'))

    all_classes = pd.read_csv(train_csv_path)['class'].unique()
    test_example_count = len(pd.read_csv(test_csv_path))-1
    print('Done: CSV parsed Unique classes: ',all_classes)

    # num_steps = 15000

    generate_tf_record = ''

    config_save_path = os.path.join(to_save_path , 'config/')


    os.makedirs(config_save_path, exist_ok=True)

    labelmap_path = os.path.join(config_save_path,'labelmap.pbtxt')
    f = open(labelmap_path, "w")
    for idx, value in enumerate(all_classes):

        first_class = '''    
    if row_label == 'first':
        return 1
                '''
        all_other_class = '''
    elif row_label == 'other':
        return 2
        '''    
        label_map = '''
item {
  id: 1
  name: 'first'
}
    '''
        if idx == 0:
            to_append = first_class.replace('first',value)
        else:
            to_append = all_other_class.replace('other',value).replace('2',str(idx+1))

        generate_tf_record+=to_append

        f.write(label_map.replace('first',value).replace('1',str(idx+1)))

    f.close()

    #print('\nGenerated string:\n',generate_tf_record)
    static_files_tf_record = os.path.join(base_config_path,"static_files/generate_tfrecord_base.py") 
    static_files_config = os.path.join(base_config_path,"static_files/faster_rcnn_inception_v2_pets_base.config") 


    with open(static_files_tf_record) as f:
        lines = f.readlines()

    custom_tf_record_path = os.path.join(config_save_path,'generate_tfrecord.py')
    with open(custom_tf_record_path, "w") as f:
        lines.insert(32, generate_tf_record)
        f.write("".join(lines))

    print('Done: Edited generate_tfrecord.py')
    print('Done: Edited labelmap.pbtxt')

    with open(static_files_config) as f:
        lines = f.readlines()

    custom_config_path = os.path.join(config_save_path, "faster_rcnn_inception_v2_pets.config")

    train_record_path = os.path.join(to_save_path,'data/train.record')
    test_record_path = os.path.join(to_save_path,'data/test.record')
    pretrained_model_path = os.path.join(base_config_path,'faster_rcnn_inception_v2_coco_2018_01_28/model.ckpt')

    with open(custom_config_path, "w") as f:
        lines[8] = '    num_classes: {}\n'.format(str(len(all_classes)))
        lines[131] = '  num_examples: {}\n'.format(str(test_example_count))
        lines[115]= '  num_steps: {}\n'.format(str(num_steps))
        lines[125]= '    input_path: "{}"\n'.format(str(train_record_path))
        lines[127]= '  label_map_path: "{}"\n'.format(str(labelmap_path))
        lines[139]= '    input_path: "{}"\n'.format(str(test_record_path))
        lines[141]= '  label_map_path: "{}"\n'.format(str(labelmap_path))
        lines[109]= '  fine_tune_checkpoint: "{}"\n'.format(str(pretrained_model_path))


        f.write("".join(lines))
        
    print('Done: Edited training/faster_rcnn_inception_v2_pets.config')
    return(str(custom_config_path))


def generate_tfrecords(base_config_path, to_save_path, dataset_path):
    config_save_path = os.path.join(to_save_path ,'config/')
    training_save_path = os.path.join(to_save_path,'training')
    os.makedirs(config_save_path, exist_ok = True)
    os.makedirs(training_save_path, exist_ok = True)

    train_csv_path = os.path.join(to_save_path,('data/train_labels.csv'))
    test_csv_path = os.path.join(to_save_path,('data/test_labels.csv'))
    custom_tf_record_path = os.path.join(config_save_path,'generate_tfrecord.py')
    # print(custom_tf_record_path)
    train_images_path = os.path.join(dataset_path,'train') 
    test_images_path = os.path.join(dataset_path,'test') 
    train_record_path = os.path.join(to_save_path,'data/train.record')
    test_record_path = os.path.join(to_save_path,'data/test.record')

    process1 = subprocess.Popen("python {} --csv_input={} --image_dir={} --output_path={}".format(custom_tf_record_path, train_csv_path, train_images_path, train_record_path), shell=True)
    process1.wait()

    process2 = subprocess.Popen("python {} --csv_input={} --image_dir={} --output_path={}".format(custom_tf_record_path, test_csv_path, test_images_path, test_record_path), shell=True) 
    process2.wait()

    print('Done: TF Records')


