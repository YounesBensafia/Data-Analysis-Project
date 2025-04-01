import os
import glob

def limit_images_in_folders(main_folder, max_images=5):

    subfolders = [f.path for f in os.scandir(main_folder) if f.is_dir()]

    for subfolder in subfolders:
        images = sorted(glob.glob(os.path.join(subfolder, "*.*")))  
        
    
        if len(images) > max_images:
            images_to_delete = images[max_images:]
            for image in images_to_delete:
                os.remove(image)
                print(f"Deleted: {image}")

if __name__ == "__main__":
    main_folder = "./Celebrity_Faces_Dataset" 
    limit_images_in_folders(main_folder)
