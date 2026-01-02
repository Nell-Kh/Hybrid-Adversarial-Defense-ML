import os

def reorganize_val_folder():
    # Path to the validation folder
    val_dir = "../data/tiny-imagenet-200/val"
    img_dir = os.path.join(val_dir, "images")
    annot_file = os.path.join(val_dir, "val_annotations.txt")
    
    # Safety check: If we already fixed it, the 'images' folder won't be there
    if not os.path.exists(img_dir):
        print("Validation folder seems already organized. Skipping.")
        return

    print("Reorganizing validation folder (this takes 10 seconds)...")
    
    # Read the text file that says which image belongs to which class
    with open(annot_file, 'r') as f:
        lines = f.readlines()
        
    for line in lines:
        parts = line.strip().split('\t')
        filename = parts[0]
        class_id = parts[1]
        
        # 1. Create the class folder if it doesn't exist
        class_folder = os.path.join(val_dir, class_id)
        if not os.path.exists(class_folder):
            os.makedirs(class_folder)
            
        # 2. Move the image into the class folder
        src = os.path.join(img_dir, filename)
        dst = os.path.join(class_folder, filename)
        if os.path.exists(src):
            os.rename(src, dst)
            
    # 3. Remove the now empty 'images' folder
    try:
        os.rmdir(img_dir)
    except:
        pass
        
    print("Done! Validation data is now ready.")

if __name__ == "__main__":
    reorganize_val_folder()