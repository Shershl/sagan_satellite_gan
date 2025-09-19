import os
import csv

root_dir = "C:/artificial intelligent systems/Dataset/military"
output_csv = "annotations.csv"

with open(output_csv, "w", newline="") as csvfile:
    writer = csv.writer(csvfile)
    writer.writerow(["image_path", "object_type"])

    for class_idx, class_name in enumerate(sorted(os.listdir(root_dir))):
        class_path = os.path.join(root_dir, class_name)
        if not os.path.isdir(class_path):
            continue
        for fname in os.listdir(class_path):
            if fname.lower().endswith((".jpg", ".png", ".jpeg")):
                image_path = os.path.join(class_path, fname).replace("\\", "/")
                writer.writerow([image_path, class_idx])

print(f"✅ Збережено annotations.csv ({output_csv})")
