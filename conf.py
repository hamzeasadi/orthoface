import os





# paths
root_data = os.path.join(os.getcwd(), 'data')

paths = dict(
    root=root_data, model=os.path.join(root_data, 'model'), face=os.path.join(root_data, 'img_align_celeba'), 
    face_id=os.path.join(root_data, 'Anno', 'identity_CelebA.txt'), face_bbox=os.path.join(root_data, 'Anno', 'list_bbox_celeba.txt')
)





def main():
    print(root_data)



if __name__ =='__main__':
    main()